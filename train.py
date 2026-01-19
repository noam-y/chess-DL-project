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

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# --- 2. Dataset Definition ---
class ChessPatchesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        
        # חיפוש CSV באופן רקורסיבי
        csv_files = glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)
        print(f"Found {len(csv_files)} CSV files in {root_dir}")

        dataframes = []
        for csv_path in csv_files:
            try:
                # הנחה: מבנה התיקיות הוא כמו ב-Colab
                # אם בקלאסטר המבנה שונה, יש להתאים את השורה הבאה
                game_folder = os.path.dirname(csv_path)
                images_dir = os.path.join(game_folder, 'tagged_images') 
                
                if not os.path.exists(images_dir):
                    continue

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if 'from_frame' in df.columns and 'fen' in df.columns:
                    df['image_dir_path'] = images_dir
                    dataframes.append(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        if dataframes:
            self.full_df = pd.concat(dataframes, ignore_index=True)
            print(f"Total frames: {len(self.full_df)}")
        else:
            self.full_df = pd.DataFrame()

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

            # Preprocessing
            image = self.resize(image)
            image = self.to_tensor(image) # (3, 480, 480)

            # Cutting to patches
            patch_size = 60
            patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
            
            labels = label_board.view(-1)
            
            return patches, labels

        except Exception as e:
            print(f"Error loading index {idx}: {e}")
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
        x = self.fc(x)
        return x

# --- 4. Main Training Function ---
def main(args):
    # הגדרת Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # Open debug file
    debug_file = open("debug.txt", "w")

    # יצירת התיקייה לשמירת המודלים אם אינה קיימת
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset & DataLoader
    dataset = ChessPatchesDataset(root_dir=args.data_dir)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_skip_none,
        num_workers=4 # יעיל יותר בקלאסטר
    )

    # Model, Loss, Optimizer
    model = PieceClassifier(num_classes=13).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    id_to_piece = {0: 'empty', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K', 7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'}
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        first_preds = None
        first_targets = None
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_data in loop:
            if batch_data is None:
                continue
            
            # Unpacking (התיקון החשוב)
            boards, labels = batch_data
            
            # Reshape for training
            inputs = boards.view(-1, 3, 60, 60).to(device)
            targets = labels.view(-1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            really_correct_right_now = 0
            for i in range(targets.size(0)):
                if predicted[i] == targets[i]:
                    really_correct_right_now += 1
                else:
                    print(f"Mismatch at index {i}: predicted {predicted[i].item()}, target {targets[i].item()}", file=debug_file)
            correct_right_now = 0
            correct_right_now = (predicted == targets).sum().item()
            print(f"total: {targets.size(0)}, correct_right_now: {correct_right_now}, really_correct_right_now: {really_correct_right_now}")
            correct += (predicted == targets).sum().item()
            
            if first_preds is None and predicted.size(0) >= 5:
                first_preds = predicted[:5].cpu().numpy()
                first_targets = targets[:5].cpu().numpy()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        # סוף אפוק - הדפסה ושמירה
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1} Summary: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")
        
        if first_preds is not None and first_targets is not None:
            print("First 5 predictions and targets:")
            print(f"Epoch {epoch+1} First 5 predictions and targets:", file=debug_file)
            for i in range(5):
                pred_name = id_to_piece.get(first_preds[i], 'unknown')
                target_name = id_to_piece.get(first_targets[i], 'unknown')
                print(f"  Pred: {pred_name}, Target: {target_name}")
                print(f"  Pred: {pred_name}, Target: {target_name}", file=debug_file)
        
        # שמירת המודל בכל אפוק (או רק בסוף)
        save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    debug_file.close()
    # הגדרת הפרמטרים שהסקריפט יודע לקבל מבחוץ
    parser = argparse.ArgumentParser(description="Train Chess Piece Classifier")
    
    # נתיבים
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the labeled_data folder")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to save the model")
    
    # היפר-פרמטרים
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of boards)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    
    main(args)