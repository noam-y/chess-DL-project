nirnir
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate
from torchvision import transforms
from PIL import Image
from chess_model_protocol import ChessModelProtocol

class BaseChessDataset(Dataset):
    def __init__(self, root_dir, mode, val_game_name, fen_converter):
        self.fen_converter = fen_converter
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
            
            # Use the passed converter function
            labels_tuple = self.fen_converter(row['fen'])
            
            patches = []
            # We need to collect labels for each patch. 
            # labels_tuple contains 8x8 tensors.
            # We will create lists for each element in the tuple.
            patch_labels = [[] for _ in range(len(labels_tuple))]

            for r in range(8):
                for c in range(8):
                    patch = self.crop_square(image, r, c)
                    patch = self.resize_transform(patch)
                    if self.transform: patch = self.transform(patch)
                    patches.append(patch)
                    
                    for i, label_tensor in enumerate(labels_tuple):
                        patch_labels[i].append(label_tensor[r, c])
            
            # Stack patches: (64, 3, 96, 96)
            patches_tensor = torch.stack(patches)
            
            # Stack labels: tuple of (64,) tensors
            stacked_labels = tuple(torch.stack(pl) for pl in patch_labels)
            
            return (patches_tensor,) + stacked_labels
        except:
            return None

class BaseChessModel(ChessModelProtocol):
    """
    Base implementation that uses the shared dataset logic.
    Subclasses only need to implement:
    - create_model
    - compute_loss
    - get_optimizer
    - infer_tile
    - fen_to_labels
    """
    def create_dataset(self, root_dir, mode='train', val_game_name='') -> Dataset:
        return BaseChessDataset(root_dir, mode, val_game_name, self.fen_to_labels)
