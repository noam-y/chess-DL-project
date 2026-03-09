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

# --- 1. Utility Functions ---

def fen_to_tensor(fen_string):
    piece_to_id = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    board_tensor = np.zeros((8, 8), dtype=np.int64)
    board_state = fen_string.split(' ')[0]
    rows = board_state.split('/')
    for r, row_str in enumerate(rows):
        c = 0
        for char in row_str:
            if char.isdigit():
                c += int(char)
            else:
                if char in piece_to_id:
                    board_tensor[r, c] = piece_to_id[char]
                c += 1
    return torch.from_numpy(board_tensor)

def get_multi_labels(labels_tensor):
    """
    - Occupancy: 0 empty, 1 occupied
    - Color: 0 (empty), 1 white , 2 black
    - Type: 0 (empty), 1-6 (P, N, B, R, Q, K)
    """
    occ = (labels_tensor > 0).long()
    
    color = torch.zeros_like(labels_tensor)
    color[(labels_tensor >= 1) & (labels_tensor <= 6)] = 1 # לבן
    color[labels_tensor >= 7] = 2 # שחור
    
    piece_type = torch.zeros_like(labels_tensor)
    mask = labels_tensor > 0
    piece_type[mask] = ((labels_tensor[mask] - 1) % 6) + 1
    
    return occ, color, piece_type

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    return default_collate(batch)

# --- 2. Dataset Definition ---

class ChessPatchesDataset(Dataset):
    def __init__(self, root_dir, game_files, transform=None):
        self.transform = transform
        dataframes = []
        for csv_path in game_files:
            try:
                game_folder = os.path.dirname(csv_path)
                images_dir = os.path.join(game_folder, 'tagged_images') 
                if not os.path.exists(images_dir): continue

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if 'from_frame' in df.columns and 'fen' in df.columns:
                    df['image_dir_path'] = images_dir
                    dataframes.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        self.full_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        self.resize = transforms.Resize((480, 480))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        try:
            row = self.full_df.iloc[idx]
            img_dir = row['image_dir_path']
            frame_id = int(row['from_frame'])
            img_name = f"frame_{frame_id:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            
            image = Image.open(img_path).convert("RGB")
            label_board = fen_to_tensor(row['fen'])

            image = self.resize(image)
            image = self.to_tensor(image)

            # splitting into 60x60 patches
            patches = image.unfold(1, 60, 60).unfold(2, 60, 60)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, 60, 60)
            labels = label_board.view(-1)
            
            return patches, labels
        except:
            return None

# --- 3. Multi-Head Model Definition ---

class MultiHeadPieceClassifier(nn.Module):
    def __init__(self):
        super(MultiHeadPieceClassifier, self).__init__()
        # Shared Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_dim = 128 * 7 * 7
        
        # Head 1: Occupancy (2 classes)
        self.head_occ = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        # Head 2: Color (3 classes)
        self.head_color = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 3))
        # Head 3: Type (7 classes)
        self.head_type = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):
        features = self.backbone(x)
        return self.head_occ(features), self.head_color(features), self.head_type(features)

# --- 4. Training Function ---

def train_one_fold(fold_idx, train_files, val_files, args, device):
    print(f"\n--- Fold {fold_idx+1} | Val Game: {os.path.basename(val_files[0])} ---")
    
    train_loader = DataLoader(ChessPatchesDataset(args.data_dir, train_files), 
                              batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(ChessPatchesDataset(args.data_dir, val_files), 
                            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_skip_none)

    model = MultiHeadPieceClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in loop:
            if batch is None: continue
            inputs, labels = batch[0].view(-1, 3, 60, 60).to(device), batch[1].view(-1).to(device)
            t_occ, t_color, t_type = get_multi_labels(labels)

            optimizer.zero_grad()
            p_occ, p_color, p_type = model(inputs)
            
            # Loss combined from all heads
            loss = criterion(p_occ, t_occ) + criterion(p_color, t_color) + criterion(p_type, t_type)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation (Checking combined accuracy)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                inputs, labels = batch[0].view(-1, 3, 60, 60).to(device), batch[1].view(-1).to(device)
                p_occ, p_color, p_type = model(inputs)
                
                res_occ = torch.argmax(p_occ, 1) == get_multi_labels(labels)[0]
                res_color = torch.argmax(p_color, 1) == get_multi_labels(labels)[1]
                res_type = torch.argmax(p_type, 1) == get_multi_labels(labels)[2]
                
                correct += (res_occ & res_color & res_type).sum().item()
                total += labels.size(0)
        
        val_acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Val Combined Acc = {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_multihead_fold{fold_idx+1}.pth"))

    return best_val_acc


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
    results = []

    for i in range(len(all_csvs)):
        val_files = [all_csvs[i]]
        train_files = [all_csvs[j] for j in range(len(all_csvs)) if i != j]
        results.append(train_one_fold(i, train_files, val_files, args, device))

    print(f"\nFinal K-Fold Mean Acc: {np.mean(results):.2f}%")

if __name__ == "__main__":
    main()