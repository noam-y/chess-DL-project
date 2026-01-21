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
import torchvision.models as models
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
    if len(batch) == 0: return None
    return default_collate(batch)

# --- 2. Dataset (Dynamic Split) ---
class SmartChessDataset(Dataset):
    def __init__(self, root_dir, mode='train', val_game_name=''):
        self.data = []
        abs_root = os.path.abspath(root_dir)
        csv_files = glob.glob(os.path.join(abs_root, '**', '*.csv'), recursive=True)
        dataframes = []
        
        self.found_games = set()

        for csv_path in csv_files:
            path_parts = csv_path.split(os.sep)
            current_game = next((part for part in path_parts if 'game' in part.lower()), None)
            
            if current_game:
                self.found_games.add(current_game)
                
                # === K fold logic
                is_val_target = (current_game == val_game_name)
                
                if mode == 'train':
                    if is_val_target: continue
                elif mode == 'val':
                    if not is_val_target: continue
            
            try:
                game_folder = os.path.dirname(csv_path)
                images_dir = os.path.join(game_folder, 'tagged_images') 
                if not os.path.exists(images_dir): 
                    images_dir = game_folder # Fallback

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if 'from_frame' in df.columns and 'fen' in df.columns:
                    df['image_dir_path'] = images_dir
                    dataframes.append(df)
            except:
                continue

        if dataframes:
            self.full_df = pd.concat(dataframes, ignore_index=True)
        else:
            self.full_df = pd.DataFrame()

        self.target_size = 96
        self.resize_transform = transforms.Resize((self.target_size, self.target_size))

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def crop_square(self, image, row, col):
        width, height = image.size
        square_w = width / 8
        square_h = height / 8
        center_x = col * square_w + square_w / 2
        center_y = row * square_h + square_h / 2
        crop_size = square_w * 1.5 
        x1 = max(0, center_x - crop_size / 2)
        y1 = max(0, center_y - crop_size / 2)
        x2 = min(width, center_x + crop_size / 2)
        y2 = min(height, center_y + crop_size / 2)
        return image.crop((x1, y1, x2, y2))

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        try:
            row = self.full_df.iloc[idx]
            base_name = f"frame_{int(row['from_frame']):06d}"
            img_path = None
            for ext in ['.jpg', '.JPG', '.jpeg', '.png']:
                temp = os.path.join(row['image_dir_path'], base_name + ext)
                if os.path.exists(temp):
                    img_path = temp
                    break
            if img_path is None: return None

            image = Image.open(img_path).convert("RGB")
            label_board = fen_to_tensor(row['fen'])
            patches = []
            labels = []
            for r in range(8):
                for c in range(8):
                    patch = self.crop_square(image, r, c)
                    patch = self.resize_transform(patch)
                    if self.transform: patch = self.transform(patch)
                    patches.append(patch)
                    labels.append(label_board[r, c])
            return torch.stack(patches), torch.stack(labels)
        except:
            return None

# --- 3. Model ---
class SmartChessNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNet, self).__init__()
        try:
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except:
            self.base_model = models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

# --- 4. Single Fold Training Function ---
def train_one_fold(args, val_game, device):
    print(f"\n{'='*40}")
    print(f"STARTING FOLD: Validate on {val_game}")
    print(f"{'='*40}")

    train_ds = SmartChessDataset(args.data_dir, mode='train', val_game_name=val_game)
    val_ds = SmartChessDataset(args.data_dir, mode='val', val_game_name=val_game)
    
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_skip_none, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn_skip_none, num_workers=4)

    model = SmartChessNet().to(device)
    
    class_weights = torch.ones(13).to(device)
    class_weights[0] = 0.2 
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.base_model.fc.parameters(), lr=0.001) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_fold_acc = 0.0

    for epoch in range(args.epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Fold {val_game} Epoch {epoch+1}", leave=False):
            if batch is None: continue
            boards, labels = batch
            inputs = boards.view(-1, 3, 96, 96).to(device)
            targets = labels.view(-1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (preds == targets).sum().item()

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                boards, labels = batch
                inputs = boards.view(-1, 3, 96, 96).to(device)
                targets = labels.view(-1).to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (preds == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        scheduler.step(epoch_loss)
        
        if val_acc > best_fold_acc:
            best_fold_acc = val_acc

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.3f}, Train Acc={train_acc:.1f}%, Val Acc ({val_game})={val_acc:.1f}%")

    print(f"Finished Fold {val_game}. Best Val Acc: {best_fold_acc:.2f}%")
    return best_fold_acc

# --- 5. Main K-Fold Loop ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    dummy_ds = SmartChessDataset(args.data_dir)
    all_games = sorted(list(dummy_ds.found_games))
    
    if not all_games:
        print("Warning: Auto-detection of games failed. Using default list.")
        all_games = ['game2', 'game7', 'game4', 'game5', 'game6']

    print(f"Found games for Cross-Validation: {all_games}")
    
    results = {}
    
    for game in all_games:
        acc = train_one_fold(args, game, device)
        results[game] = acc
    
    print("\n" + "="*40)
    print("FINAL K-FOLD RESULTS")
    print("="*40)
    accuracies = []
    for game, acc in results.items():
        print(f"Hold-out {game}: {acc:.2f}%")
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("-" * 40)
    print(f"Average Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=5) # epochs per fold
    parser.add_argument("--batch_size", type=int, default=16) 
    args = parser.parse_args()
    main(args)