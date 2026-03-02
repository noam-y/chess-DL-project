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

def fen_to_targets(fen_string):
    type_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    
    occ_tensor = np.zeros((8, 8), dtype=np.int64)
    color_tensor = np.zeros((8, 8), dtype=np.int64)
    piece_tensor = np.zeros((8, 8), dtype=np.int64)
    
    board_state = fen_string.split(' ')[0]
    rows = board_state.split('/')
    
    for r, row_str in enumerate(rows):
        c = 0
        for char in row_str:
            if char.isdigit():
                c += int(char)
            else:
                occ_tensor[r, c] = 1
                color_tensor[r, c] = 1 if char.isupper() else 0
                piece_tensor[r, c] = type_map[char.lower()]
                c += 1
                
    return torch.from_numpy(occ_tensor), torch.from_numpy(color_tensor), torch.from_numpy(piece_tensor)

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    return default_collate(batch)

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
                is_val_target = (current_game == val_game_name)

                if mode == 'train' and is_val_target: continue
                elif mode == 'val' and not is_val_target: continue

            try:
                game_folder = os.path.dirname(csv_path)
                images_dir = os.path.join(game_folder, 'tagged_images')
                if not os.path.exists(images_dir):
                    images_dir = game_folder 

                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if 'from_frame' in df.columns and 'fen' in df.columns:
                    df['image_dir_path'] = images_dir
                    dataframes.append(df)
            except:
                continue

        self.full_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
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
            t_occ, t_color, t_piece = fen_to_targets(row['fen'])
            
            patches, l_occ, l_color, l_piece = [], [], [], []
            for r in range(8):
                for c in range(8):
                    patch = self.crop_square(image, r, c)
                    patch = self.resize_transform(patch)
                    if self.transform: patch = self.transform(patch)
                    patches.append(patch)
                    l_occ.append(t_occ[r, c])
                    l_color.append(t_color[r, c])
                    l_piece.append(t_piece[r, c])
                    
            return (torch.stack(patches), 
                    torch.stack(l_occ), 
                    torch.stack(l_color), 
                    torch.stack(l_piece))
        except:
            return None

class SmartChessNetV3(nn.Module):
    def __init__(self):
        super(SmartChessNetV3, self).__init__()
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except:
            resnet = models.resnet18(pretrained=True)
            
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        
        self.head_occ = nn.Linear(num_ftrs, 2)
        self.head_color = nn.Linear(num_ftrs, 2)
        self.head_piece = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        
        out_occ = self.head_occ(features)
        out_color = self.head_color(features)
        out_piece = self.head_piece(features)
        
        return out_occ, out_color, out_piece

def train_one_fold(args, val_game, device):
    print(f"\nSTARTING FOLD: Validate on {val_game}")
    train_ds = SmartChessDataset(args.data_dir, mode='train', val_game_name=val_game)
    val_ds = SmartChessDataset(args.data_dir, mode='val', val_game_name=val_game)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_skip_none, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn_skip_none, num_workers=4)

    model = SmartChessNetV3().to(device)

    weight_occ = torch.tensor([0.722, 1.624], dtype=torch.float32).to(device)
    weight_color = torch.tensor([1.026, 0.976], dtype=torch.float32).to(device)
    weight_piece = torch.tensor([0.319, 1.898, 1.676, 1.222, 3.133, 1.642], dtype=torch.float32).to(device)

    criterion_occ = nn.CrossEntropyLoss(weight=weight_occ)
    criterion_color = nn.CrossEntropyLoss(weight=weight_color)
    criterion_piece = nn.CrossEntropyLoss(weight=weight_piece)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_fold_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Fold {val_game} Epoch {epoch+1}", leave=False):
            if batch is None: continue
            boards, l_occ, l_color, l_piece = batch
            
            inputs = boards.view(-1, 3, 96, 96).to(device)
            t_occ = l_occ.view(-1).to(device)
            t_color = l_color.view(-1).to(device)
            t_piece = l_piece.view(-1).to(device)

            optimizer.zero_grad()
            out_occ, out_color, out_piece = model(inputs)
            
            loss_occ = criterion_occ(out_occ, t_occ)
            
            mask = (t_occ == 1)
            loss_color = criterion_color(out_color[mask], t_color[mask]) if mask.sum() > 0 else 0
            loss_piece = criterion_piece(out_piece[mask], t_piece[mask]) if mask.sum() > 0 else 0
            
            loss = loss_occ + loss_color + loss_piece
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_correct_full = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                boards, l_occ, l_color, l_piece = batch
                inputs = boards.view(-1, 3, 96, 96).to(device)
                
                t_occ = l_occ.view(-1).to(device)
                t_color = l_color.view(-1).to(device)
                t_piece = l_piece.view(-1).to(device)
                
                out_occ, out_color, out_piece = model(inputs)
                
                pred_occ = torch.argmax(out_occ, 1)
                pred_color = torch.argmax(out_color, 1)
                pred_piece = torch.argmax(out_piece, 1)
                
                correct_empty = (pred_occ == 0) & (t_occ == 0)
                correct_occupied = (pred_occ == 1) & (t_occ == 1) & (pred_color == t_color) & (pred_piece == t_piece)
                
                correct_squares = correct_empty | correct_occupied
                
                val_total += t_occ.size(0)
                val_correct_full += correct_squares.sum().item()

        epoch_loss = running_loss / len(train_loader)
        val_acc = 100 * val_correct_full / val_total if val_total > 0 else 0

        scheduler.step(epoch_loss)

        if val_acc > best_fold_acc:
            best_fold_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{val_game}.pth"))

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.3f}, Hierarchical Val Acc={val_acc:.2f}%")

    return best_fold_acc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dummy_ds = SmartChessDataset(args.data_dir)
    all_games = sorted(list(dummy_ds.found_games))

    results = {}
    for game in all_games:
        acc = train_one_fold(args, game, device)
        results[game] = acc

    accuracies = list(results.values())
    print(f"\nFinal LOGO CV Average Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_v3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
