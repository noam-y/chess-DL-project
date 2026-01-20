import os
import glob
import random
import sys
from enum import Enum
from pathlib import Path

import chess
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision import transforms, models
import pytorch_lightning as pl

# -----------------------------------------------------------------------------
# Board & Helper Classes
# -----------------------------------------------------------------------------

class Board:
    def __init__(self, board):
        self.board = board

    @classmethod
    def from_fen(cls, fen):
        try:
            # Basic validation/cleanup if needed
            board = chess.Board(fen=fen)
        except ValueError:
            print(f"Warning: Could not parse FEN: {fen}")
            board = chess.Board() # Default starting position
        return Board(board)

    @classmethod
    def from_file_name(cls, file_name):
        # Legacy support or if needed
        try:
            fen = Path(file_name).stem
            fen = fen.replace("-", "/") + " w KQkq - 0 1"
            board = chess.Board(fen=fen)
        except ValueError:
            print(f"Warning: Could not parse FEN from filename {file_name}")
            board = chess.Board()
        return Board(board)

    @classmethod
    def from_array(cls, a):
        board = chess.Board()
        board.clear()
        for file in range(8):
            for rank in range(8):
                i = a[rank][file]
                if i == 0:
                    continue
                sq = (rank * 8) + file
                piece = Board.piece_from_int(i)
                board.set_piece_at(sq, piece)
        return Board(board)

    @classmethod
    def piece_to_int(cls, piece):
        if piece is None:
            return 0
        return piece.piece_type if piece.color else piece.piece_type + 6

    @classmethod
    def piece_from_int(cls, i):
        if i == 0:
            return None
        piece_type = ((i-1)%6)+1
        piece_color = chess.BLACK if i > 6 else chess.WHITE
        return chess.Piece(piece_type=piece_type, color=piece_color)

    def to_array(self):
        a = np.zeros((8,8), dtype=np.int8)
        for sq, piece in self.board.piece_map().items():
            file = sq % 8
            rank = sq // 8
            a[rank][file] = Board.piece_to_int(piece)
        return a

    def flip(self):
        cp = np.copy(self.to_array())
        flipped = np.fliplr(cp)
        return Board.from_array(flipped)

class PieceSet(Enum):
    # empty / piece
    Binary = 'binary'
    # empty / white / black
    Colors = 'colors'
    # empty / pawn / knight / bishop / rook / queen / king
    ColorBlind = 'color-blind'
    # empty / white pawn / ... / black king
    Full = 'full'

    def weights(self):
        if self == PieceSet.Binary:
            return torch.tensor([1.,1.])
        elif self == PieceSet.Colors:
            return torch.tensor([1.,2.,2.])
        elif self == PieceSet.ColorBlind:
            return torch.tensor([64/32,64/16,64/4,64/4,64/4,64/2,64/2])
        elif self == PieceSet.Full:
            return torch.tensor([64/32,64/8,64/2,64/2,64/2,64/1,64/1,64/8,64/2,64/2,64/2,64/1,64/1])

    def mapping(self):
        if self == PieceSet.Binary:
            return {"PNBRQKpnbrqk":1}
        elif self == PieceSet.Colors:
            return {"PNBRQK":1,"pnbrqk":2}
        elif self == PieceSet.ColorBlind:
            return {"Pp":1,"Nn":2,"Bb":3,"Rr":4,"Qq":5,"Kk":6}
        elif self == PieceSet.Full:
            return {"P":1,"N":2,"B":3,"R":4,"Q":5,"K":6,"p":7,"n":8,"b":9,"r":10,"q":11,"k":12}
        return {}

    def transform_y(self, y):
        for symbol, value in self.mapping().items():
            for char in symbol:
                piece = chess.Piece.from_symbol(char)
                i = Board.piece_to_int(piece)
                y = np.where(y == i, value, y)
        return y

    def transform_y_hat(self, y_hat):
        empty = torch.unsqueeze(torch.sum(y_hat[:,0:1],1),1)
        elements = [empty]
        for symbol, value in self.mapping().items():
            element = None
            for char in symbol:
                piece = chess.Piece.from_symbol(char)
                i = Board.piece_to_int(piece)
                if element is None:
                    element = y_hat[:,i:i+1]
                else:
                    element = element + y_hat[:,i:i+1]
            elements.append(element)
        return torch.cat(elements, 1)

# -----------------------------------------------------------------------------
# Dataset & Transforms
# -----------------------------------------------------------------------------

def load_samples_from_dir(root_dir):
    """
    Crawls root_dir for CSV files, expects 'tagged_images' folder nearby.
    Returns list of dicts: {'image_path': str, 'fen': str}
    """
    samples = []
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)
    print(f"Found {len(csv_files)} CSV files in {root_dir}")

    for csv_path in csv_files:
        try:
            game_folder = os.path.dirname(csv_path)
            images_dir = os.path.join(game_folder, 'tagged_images')
            
            if not os.path.exists(images_dir):
                continue

            df = pd.read_csv(csv_path)
            # Normalize columns
            df.columns = df.columns.str.strip()
            
            if 'from_frame' in df.columns and 'fen' in df.columns:
                for _, row in df.iterrows():
                    frame_id = int(row['from_frame'])
                    fen = row['fen']
                    # Assuming format frame_000001.jpg based on train.py
                    img_name = f"frame_{frame_id:06d}.jpg"
                    img_path = os.path.join(images_dir, img_name)
                    
                    # Only add if file exists
                    if os.path.exists(img_path):
                        samples.append({
                            'image_path': img_path,
                            'fen': fen
                        })
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            
    return samples

class BoardDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data = self.samples[idx]
        image_path = sample_data['image_path']
        fen = sample_data['fen']
        
        image = Image.open(image_path).convert("RGB")
        board = Board.from_fen(fen)
        
        sample = {'image': image, 'board': board}
        if self.transform:
            sample = self.transform(sample)
        return sample

class FlipTransform(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        image, board = sample['image'], sample['board']
        p = random.uniform(0, 1)
        flip = p <= self.probability
        if flip:
            image = ImageOps.mirror(image)
            board = board.flip()
        return {'image': image, 'board': board}

class ImageTransform(object):
    def __init__(self, f=None):
        self.f = f

    def __call__(self, sample):
        image, board = sample['image'], sample['board']
        if not self.f is None:
            image = self.f(image)
        return {'image': image, 'board': board}

class TensorTransform(object):
    def __init__(self, piece_sets=[PieceSet.Binary, PieceSet.Colors, PieceSet.ColorBlind, PieceSet.Full]):
        self.piece_sets = piece_sets
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample):
        image, board = sample['image'], sample['board']
        t = self.preprocess(image.copy())
        result = {"image":t}
        board_arr = board.to_array().flatten().astype(int)
        for pc in self.piece_sets:
            result[pc.value] = pc.transform_y(board_arr)
        return result

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class Model(pl.LightningModule):
    def __init__(self, resnet=None, piece_sets=[PieceSet.Binary, PieceSet.Colors, PieceSet.ColorBlind, PieceSet.Full],
                 lr=0.0001, batch_size=8):
        super().__init__()
        self.piece_sets = piece_sets
        self.lr = lr
        self.batch_size = batch_size
        self.resnet = resnet
        # EfficientNet V2 S output is 1280, but the notebook uses 1000? 
        # Check notebook usage: `models.efficientnet_v2_s(weights=...)`
        # By default efficientnet_v2_s classifier[1] is Linear(1280, 1000).
        # So output is 1000.
        self.outputs = nn.Linear(1000, 64*13)
        self.losses = nn.ModuleList()
        for pc in piece_sets:
            loss = nn.CrossEntropyLoss(weight=pc.weights())
            self.losses.append(loss)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = self.outputs(x)
        x = torch.reshape(x, (x.shape[0], 64, 13))
        return F.softmax(x,dim=2)

    def training_step(self, batch, batch_idx):
        result = self.combined_loss(batch)
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        return self.combined_loss(batch, prefix="val_")

    def combined_loss(self, batch, prefix=""):
        x = batch['image']
        output = self(x)
        loss = torch.tensor([0.]).type_as(x)
        results = {}
        for i, pc in enumerate(self.piece_sets):
            # loss
            y = batch[pc.value]
            y = torch.flatten(y, end_dim=1)
            y_hat = output
            y_hat = torch.flatten(y_hat, end_dim=1)
            y_hat = pc.transform_y_hat(y_hat)
            pc_loss = self.losses[i](y_hat, y) 
            loss += pc_loss
            # accuracy
            prediction = torch.argmax(y_hat, dim=1)
            correct = torch.sum((y == prediction).float())
            accuracy = correct / prediction.shape[0]
            self.log(prefix + pc.value + "_loss", pc_loss)
            self.log(prefix + pc.value + "_accuracy", accuracy, prog_bar=True)
        results[prefix +'loss'] = loss
        self.log(prefix + "loss", loss, prog_bar=True)
        return results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# -----------------------------------------------------------------------------
# Main Fine-Tuning Script
# -----------------------------------------------------------------------------

def main():
    # Configuration
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 10
    DATA_DIR = "./assets/labeled_data" # Updated path to match cluster structure
    CHECKPOINT_PATH = "2023-11-02-fenify-3d-efficientnet-v2-s-95-val-acc.ckpt" # Update if you have one

    # 1. Setup Data
    print(f"Searching for data in {DATA_DIR}...")
    all_samples = load_samples_from_dir(DATA_DIR)
    
    if not all_samples:
        print(f"No samples found in {DATA_DIR}. Please check the path and structure.")
        return

    print(f"Total samples found: {len(all_samples)}")

    # Transforms
    random_color = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)
    random_rot = T.RandomRotation(degrees=(-15,15))
    random_perspective = T.RandomPerspective(distortion_scale=0.25, p=1.0)
    random_geo = T.RandomChoice([random_rot,random_perspective])
    
    image_transform = ImageTransform(f=T.Compose([
        T.RandomApply([random_geo], p=0.33),
        T.RandomApply([random_color], p=0.25),
        T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.25),
    ]))
    
    train_transform = transforms.Compose([
        image_transform,
        FlipTransform(),
        TensorTransform(),
    ])
    
    val_transform = transforms.Compose([
        TensorTransform(),
    ])

    # Split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Training on {len(train_samples)} images, validating on {len(val_samples)} images.")

    # Datasets
    ds_train = BoardDataset(train_samples, transform=train_transform)
    ds_val = BoardDataset(val_samples, transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Setup Model
    PIECE_SETS = [PieceSet.Binary, PieceSet.Colors, PieceSet.ColorBlind, PieceSet.Full]
    
    # Initialize base model
    resnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    
    model = Model(resnet=resnet, piece_sets=PIECE_SETS, lr=LEARNING_RATE, batch_size=BATCH_SIZE)

    # Load Checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        # Handle state dict key matching if needed
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("No checkpoint found. Training from ImageNet weights.")

    # 3. Trainer
    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", name="fenify_finetune")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        accelerator="auto", # auto detect gpu/cpu
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    # 4. Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Save fine-tuned model
    trainer.save_checkpoint("gemini_fenify_finetuned.ckpt")
    print("Training complete. Model saved to gemini_fenify_finetuned.ckpt")

    # Optional: Save as JIT for inference
    model.eval()
    input_sample = torch.randn(1, 3, 400, 400)
    traced_model = torch.jit.trace(model, input_sample)
    traced_model.save("gemini_fenify_finetuned.pt")
    print("JIT model saved to gemini_fenify_finetuned.pt")

if __name__ == "__main__":
    main()
