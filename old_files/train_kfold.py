import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def fen_to_tensor(fen_string):
    piece_to_id = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
    board_tensor = np.zeros((8, 8), dtype=np.int64)
    board_state = fen_string.split(' ')[0]
    rows = board_state.split('/')
    for r, row_str in enumerate(rows):
        c = 0
        for char in row_str:
            if char.isdigit(): c += int(char)
            else:
                if char in piece_to_id: board_tensor[r, c] = piece_to_id[char]
                c += 1
    return torch.from_numpy(board_tensor)

def get_multi_labels(labels_tensor):
    occ = (labels_tensor > 0).long()
    color = torch.zeros_like(labels_tensor)
    color[(labels_tensor >= 1) & (labels_tensor <= 6)] = 1 
    color[labels_tensor >= 7] = 2 
    piece_type = torch.zeros_like(labels_tensor)
    mask = labels_tensor > 0
    piece_type[mask] = ((labels_tensor[mask] - 1) % 6) + 1
    return occ, color, piece_type

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch) if batch else None

class ChessPatchesDataset(Dataset):
    def __init__(self, root_dir, game_files):
        dataframes = []
        for csv_path in game_files:
            try:
                images_dir = os.path.join(os.path.dirname(csv_path), 'tagged_images') 
                if not os.path.exists(images_dir): continue
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if 'from_frame' in df.columns and 'fen' in df.columns:
                    df['image_dir_path'] = images_dir
                    dataframes.append(df)
            except Exception as e: print(f"Error: {e}")
        self.full_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        self.resize = transforms.Resize((480, 480))
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.full_df)

    def __getitem__(self, idx):
        try:
            row = self.full_df.iloc[idx]
            img_path = os.path.join(row['image_dir_path'], f"frame_{int(row['from_frame']):06d}.jpg")
            image = self.to_tensor(self.resize(Image.open(img_path).convert("RGB")))
            patches = image.unfold(1, 60, 60).unfold(2, 60, 60).permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, 60, 60)
            return patches, fen_to_tensor(row['fen']).view(-1)
        except: return None

class MultiHeadPieceClassifier(nn.Module):
    def __init__(self):
        super(MultiHeadPieceClassifier, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_dim = 128 * 7 * 7
        self.head_occ = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        self.head_color = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 3))
        self.head_type = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):
        features = self.backbone(x)
        return self.head_occ(features), self.head_color(features), self.head_type(features)

def train_one_fold(fold_idx, train_files, val_files, args, device):
    print(f"\n--- Fold {fold_idx+1} | Val: {os.path.basename(val_files[0])} ---")
    train_loader = DataLoader(ChessPatchesDataset(args.data_dir, train_files), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(ChessPatchesDataset(args.data_dir, val_files), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_skip_none)
    model = MultiHeadPieceClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if batch is None: continue
            inputs, labels = batch[0].view(-1, 3, 60, 60).to(device), batch[1].view(-1).to(device)
            t_occ, t_color, t_type = get_multi_labels(labels)
            optimizer.zero_grad()
            p_occ, p_color, p_type = model(inputs)
            loss = criterion(p_occ, t_occ) + criterion(p_color, t_color) + criterion(p_type, t_type)
            loss.backward(); optimizer.step()
        model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                inputs, labels = batch[0].view(-1, 3, 60, 60).to(device), batch[1].view(-1).to(device)
                p_o, p_c, p_t = model(inputs)
                m_o, m_c, m_t = get_multi_labels(labels)
                res = (torch.argmax(p_o, 1) == m_o) & (torch.argmax(p_c, 1) == m_c) & (torch.argmax(p_t, 1) == m_t)
                correct += res.sum().item(); total += labels.size(0)
        acc = 100. * correct / total if total > 0 else 0
        print(f"Val Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_multihead_fold{fold_idx+1}.pth"))
    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_multihead")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    all_csvs = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.csv'), recursive=True))
    results = [train_one_fold(i, [all_csvs[j] for j in range(len(all_csvs)) if i != j], [all_csvs[i]], args, device) for i in range(len(all_csvs))]
    print(f"\nFinal K-Fold Mean Acc: {np.mean(results):.2f}%")

if __name__ == "__main__": main()
