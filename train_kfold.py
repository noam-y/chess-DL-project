import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.metrics import classification_report, confusion_matrix
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

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# --- 2. Dataset Definition - including filtering games ---
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
                    df['source_csv'] = csv_path
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
            fen_label = row['fen']

            img_name = f"frame_{frame_id:06d}.jpg"
            img_path = os.path.join(img_dir, img_name)
            
            image = Image.open(img_path).convert("RGB")
            label_board = fen_to_tensor(fen_label)

            image = self.resize(image)
            image = self.to_tensor(image)

            patch_size = 60
            patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
            labels = label_board.view(-1)
            
            return patches, labels
        except Exception:
            return None

# --- 3. Model Definition ---
class PieceClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(PieceClassifier, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.fc_input_dim = 128 * 7 * 7
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)

# --- 4. Training Function for a single fold ---
def train_one_fold(fold_idx, train_files, val_files, args, device):
    print(f"\n{'='*30}\nStarting Fold {fold_idx+1}\n{'='*30}")
    
    train_ds = ChessPatchesDataset(args.data_dir, train_files)
    val_ds = ChessPatchesDataset(args.data_dir, val_files)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_skip_none)

    model = PieceClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    class_weights = torch.ones(13).to(device)
    class_weights[0] = 0.1
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}")
        for batch in loop:
            if batch is None: continue
            boards, labels = batch
            inputs, targets = boards.view(-1, 3, 60, 60).to(device), labels.view(-1).to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                boards, labels = batch
                inputs, targets = boards.view(-1, 3, 60, 60).to(device), labels.view(-1).to(device)
                _, predicted = torch.max(model(inputs), 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Fold {fold_idx+1} Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_fold{fold_idx+1}.pth"))

    return best_val_acc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    all_csvs = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.csv'), recursive=True))
    num_folds = len(all_csvs)
    print(f"Found {num_folds} games. Starting Leave-One-Game-Out CV...")

    fold_results = []
    for i in range(num_folds):
        val_files = [all_csvs[i]]
        train_files = [all_csvs[j] for j in range(num_folds) if i != j]
        
        res = train_one_fold(i, train_files, val_files, args, device)
        fold_results.append(res)

    print(f"\nFinal K-Fold Mean Accuracy: {np.mean(fold_results):.2f}% (+/- {np.std(fold_results):.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_kfold")
    parser.add_argument("--epochs", type=int, default=15) # to avoid overfitting
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    main(parser.parse_args())