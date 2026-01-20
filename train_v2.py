import os
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.metrics import classification_report
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

# --- 1. הגדרות ופונקציות עזר ---

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
    # סינון דוגמאות פגומות (אם לא הצלחנו לטעון תמונה)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# --- 2. הדאטה-סט המשופר ---
class SmartChessDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.data = []
        self.mode = mode
        
        # 1. תיקון נתיבים (מונע קריסות)
        abs_root = os.path.abspath(root_dir)
        print(f"Scanning: {abs_root}")
        
        csv_files = glob.glob(os.path.join(abs_root, '**', '*.csv'), recursive=True)
        
        dataframes = []
        for csv_path in csv_files:
            # 2. הפרדה חכמה: Train vs Test (Game 6)
            is_game6 = 'game6.csv' in csv_path
            
            if mode == 'train' and is_game6:
                print(f"Skipping {os.path.basename(csv_path)} (Saved for validation)")
                continue
            elif mode == 'val' and not is_game6:
                continue # ב-Validation לוקחים רק את game6

            try:
                game_folder = os.path.dirname(csv_path)
                images_dir = os.path.join(game_folder, 'tagged_images') 
                
                if not os.path.exists(images_dir): continue

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if 'from_frame' in df.columns and 'fen' in df.columns:
                    df['image_dir_path'] = images_dir
                    dataframes.append(df)
            except:
                continue

        if dataframes:
            self.full_df = pd.concat(dataframes, ignore_index=True)
            print(f"[{mode.upper()}] Total frames: {len(self.full_df)}")
        else:
            self.full_df = pd.DataFrame()

        # 3. אוגמנטציות חכמות (רק לאימון!)
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((480, 480)),
                # ColorJitter: מכריח את המודל להתעלם מתאורה ספציפית
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(5), # מדמה מצלמה עקומה
                transforms.ToTensor(),
                # נרמול לפי הסטנדרט של ImageNet (קריטי ל-ResNet)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # בטסט אנחנו רוצים תמונה נקייה
            self.transform = transforms.Compose([
                transforms.Resize((480, 480)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        try:
            row = self.full_df.iloc[idx]
            img_path = os.path.join(row['image_dir_path'], f"frame_{int(row['from_frame']):06d}.jpg")
            
            if not os.path.exists(img_path): return None

            image = Image.open(img_path).convert("RGB")
            label_board = fen_to_tensor(row['fen'])

            image = self.transform(image)

            # חיתוך למשבצות 60x60
            patch_size = 60
            patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
            
            return patches, label_board.view(-1)

        except:
            return None

# --- 3. המודל החכם (ResNet18 Customized) ---
class SmartChessNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNet, self).__init__()
        print("Loading Pre-trained ResNet18...")
        
        # טוענים מודל שכבר יודע "לראות" (ImageNet)
        try:
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except:
            self.base_model = models.resnet18(pretrained=True)
            
        # מחליפים את השכבה האחרונה כדי שתתאים ל-13 הכלים שלנו
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

# --- 4. לולאת האימון ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # יצירת דאטה-סטים
    train_ds = SmartChessDataset(args.data_dir, mode='train')
    # אם תרצי, אפשר ליצור גם val_ds כדי לבדוק ביצועים תוך כדי ריצה
    # val_ds = SmartChessDataset(args.data_dir, mode='val')

    if len(train_ds) == 0:
        print("Error: No training data found.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn_skip_none, num_workers=2)

    model = SmartChessNet().to(device)

    # 4. טיפול ב-Class Imbalance
    # נותנים פחות משקל למשבצות ריקות (0) כדי שהמודל יתאמץ לזהות כלים
    class_weights = torch.ones(13).to(device)
    class_weights[0] = 0.2 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # אופטימיזציה
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 5. Scheduler: מאיץ כשקל, מאט כשקשה
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in loop:
            if batch is None: continue
            
            # (Batch, 64, 3, 60, 60) -> (Batch*64, 3, 60, 60)
            boards, labels = batch
            inputs = boards.view(-1, 3, 60, 60).to(device)
            targets = labels.view(-1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            batch_total = targets.size(0)
            batch_correct = (preds == targets).sum().item()
            
            total += batch_total
            correct += batch_correct
            
            loop.set_postfix(loss=loss.item(), acc=batch_correct/batch_total)

        # סיכום אפוק
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # עדכון Scheduler
        scheduler.step(epoch_loss)
        
        print(f"Summary Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%")

        # שמירת המודל הטוב ביותר
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_smart_model.pth"))
            print(">>> New Best Model Saved!")
        
        # שמירה רגילה
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)