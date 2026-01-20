import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

# Adding current dir to path to avoid import errors
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}
ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}
ID_TO_PIECE[0] = '1'

# --- 1. Model Definition (Must match train.py exactly) ---
class SmartChessNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNet, self).__init__()
        # We don't need pre-trained weights here, we will load our own checkpoint
        try:
            self.base_model = models.resnet18(weights=None) 
        except:
            self.base_model = models.resnet18(pretrained=False)
            
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

# --- 2. Helper Functions ---
def prediction_to_fen(pred_tensor):
    """ Converts the 8x8 prediction tensor to a FEN string """
    board = pred_tensor.cpu().numpy()
    rows_fen = []
    for r in range(8):
        row_str = ""
        empty_count = 0
        for c in range(8):
            val = board[r, c]
            char = ID_TO_PIECE.get(val, '1')
            if char == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += char
        if empty_count > 0:
            row_str += str(empty_count)
        rows_fen.append(row_str)
    return "/".join(rows_fen)

def compare_fens(true_fen, pred_fen):
    """ Compares Ground Truth FEN with Predicted FEN """
    def expand(fen):
        rows = fen.split(' ')[0].split('/')
        res = []
        for row in rows:
            for char in row:
                if char.isdigit():
                    res.extend(['.'] * int(char))
                else:
                    res.append(char)
        return res

    list_true = expand(true_fen)
    list_pred = expand(pred_fen)
    
    if len(list_pred) != 64: return 0, False

    correct_count = sum([1 for t, p in zip(list_true, list_pred) if t == p])
    is_perfect = (correct_count == 64)
    return correct_count, is_perfect

# --- 3. Smart Dataset for Evaluation ---
class SmartEvalDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        # Handle paths safely
        self.root_dir = os.path.abspath(root_dir)
        
        # Target size for ResNet
        self.target_size = 224
        
        # 1. Resize the cropped patch to 224x224
        self.resize_transform = transforms.Resize((self.target_size, self.target_size))