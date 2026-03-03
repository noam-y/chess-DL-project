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
        self.all_labels = [] # Now stores raw strings directly from the CSV

        abs_root = os.path.abspath(root_dir)
        csv_files = glob.glob(os.path.join(abs_root, '**', 'gt.csv'), recursive=True)
        
        missing_count = 0

        for csv_path in csv_files:
            game_folder = os.path.dirname(csv_path)
            current_game = os.path.basename(game_folder)
            
            # K-fold cross-validation split logic
            is_val_target = (current_game == val_game_name)
            if mode == 'train' and is_val_target: 
                continue
            elif mode == 'val' and not is_val_target: 
                continue
        
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip() 
                
                # Determine the column name for the filename
                filename_col = 'file_name' if 'file_name' in df.columns else 'file name'
                
                if filename_col in df.columns and 'fen' in df.columns:
                    for _, row in df.iterrows():
                        img_name = str(row[filename_col]).strip()
                        fen_char = str(row['fen']).strip()
                        
                        # Check for 'tagged_images' (new dataset) or 'images' (old dataset)
                        img_path_new = os.path.join(game_folder, 'tagged_images', img_name)
                        img_path_old = os.path.join(game_folder, 'images', img_name)
                        
                        if os.path.exists(img_path_new):
                            img_path = img_path_new
                        elif os.path.exists(img_path_old):
                            img_path = img_path_old
                        else:
                            missing_count += 1
                            if missing_count <= 3:
                                print(f"Warning: Image not found: {img_path_new} (or {img_path_old})")
                            continue

                        self.data.append((img_path, fen_char))
                        # Just append the raw string from the CSV!
                        self.all_labels.append(fen_char)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        
        if missing_count > 0:
            print(f"Dataset ({mode}): Skipped {missing_count} missing images.")

        # Transforms setup...
        self.target_size = 96
        self.resize_transform = transforms.Resize((self.target_size, self.target_size))

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_path, fen_char = self.data[idx]
            
            image = Image.open(img_path).convert("RGB")
            image = self.resize_transform(image)
            if self.transform: 
                image = self.transform(image)
            
            labels_tuple = self.fen_converter(fen_char)
            
            return (image,) + labels_tuple
        except Exception as e:
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
