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

# --- 1. Utility Functions ---

def fen_to_tensor(fen_string):
    """
    Converts a FEN string (Chess notation) into an 8x8 tensor of integers.
    """
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
    """
    Filters out corrupted samples (if image loading failed) from the batch.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# --- 2. The Improved Dataset (with Contextual Cropping) ---
class SmartChessDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.data = []
        self.mode = mode
        
        # 1. Path correction (prevents crashes with relative paths)
        abs_root = os.path.abspath(root_dir)
        print(f"Scanning: {abs_root}")
        
        csv_files = glob.glob(os.path.join(abs_root, '**', '*.csv'), recursive=True)
        
        dataframes = []
        for csv_path in csv_files:
            is_game6 = 'game6' in csv_path
            
            # Logic for Training vs Validation sets
            if mode == 'train' and is_game6:
                print(f"Skipping {os.path.basename(csv_path)} (Saved for validation)")
                continue
            elif mode == 'val' and not is_game6:
                continue # In validation mode, we only want game6

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

        self.target_size = 96
        
        self.resize_transform = transforms.Resize((self.target_size, self.target_size))

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomRotation(5), # More rotation allowed due to padding
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # For testing, we want a clean image
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def crop_square(self, image, row, col):
        """
        The secret to improvement: Contextual Cropping.
        We crop a square larger than the actual grid cell to capture the 'head' 
        of tall pieces (like the Queen or King) that might be visually located 
        in the square above.
        """
        width, height = image.size
        square_w = width / 8
        square_h = height / 8
        
        center_x = col * square_w + square_w / 2
        center_y = row * square_h + square_h / 2
        
        # Factor 1.5 = taking 50% more area from each side
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
            # Support multiple file extensions
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

            # Loop over all 64 squares
            for r in range(8):
                for c in range(8):
                    patch = self.crop_square(image, r, c)
                    patch = self.resize_transform(patch)
                    
                    if self.transform:
                        patch = self.transform(patch)
                    
                    patches.append(patch)
                    labels.append(label_board[r, c])
            
            return torch.stack(patches), torch.stack(labels)

        except:
            return None

# --- 3. The Smart Model (ResNet18 Customized) ---
class SmartChessNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNet, self).__init__()
        # print("Loading Pre-trained ResNet18...")
        try:
            # Load a model that already knows how to "see" (ImageNet weights)
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except:
            self.base_model = models.resnet18(pretrained=True)
            
        # Replace the last layer to fit our 13 chess classes
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

# --- 4. Training Loop ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create Datasets
    train_ds = SmartChessDataset(args.data_dir, mode='train')

    if len(train_ds) == 0:
        print("Error: No training data found.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn_skip_none, num_workers=4)

    model = SmartChessNet().to(device)

    class_weights = torch.ones(13).to(device)
    class_weights[0] = 0.2 
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for Fine-tuning
    # Scheduler: Accelerates when easy, slows down when stuck (Plateau)
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
            
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=correct/total)

        # Epoch Summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Update Scheduler
        scheduler.step(epoch_loss)
        print(f"Summary Epoch {epoch+1}: Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%")

        # Save Best Model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_smart_model_v2.pth"))
        
        # Regular save
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_v2_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    # Default is 2 because images are larger and might fill memory
    parser.add_argument("--batch_size", type=int, default=16) 
    args = parser.parse_args()
    main(args)