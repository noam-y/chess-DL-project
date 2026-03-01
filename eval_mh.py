import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import torch.nn as nn

class MultiHeadPieceClassifier(nn.Module):
    def __init__(self):
        super(MultiHeadPieceClassifier, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_dim = 128 * 7 * 7
        self.head_occ = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        self.head_color = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 3))
        self.head_type = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):
        features = self.backbone(x)
        return self.head_occ(features), self.head_color(features), self.head_type(features)

def combine_heads_to_fen(p_occ, p_color, p_type):
    occ = torch.argmax(p_occ, 1) # 0=Empty, 1=Occupied
    color = torch.argmax(p_color, 1) # 1=White, 2=Black
    ptype = torch.argmax(p_type, 1) # 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K
    
    type_map = {1:'p', 2:'n', 3:'b', 4:'r', 5:'q', 6:'k'}
    
    res_board = []
    for i in range(len(occ)):
        if occ[i] == 0:
            res_board.append('1')
        else:
            p_char = type_map.get(ptype[i].item(), 'p')
            if color[i] == 1: # White -> Uppercase
                res_board.append(p_char.upper())
            else: # Black -> Lowercase
                res_board.append(p_char.lower())
    
    rows = []
    for r in range(8):
        row_chars = res_board[r*8 : (r+1)*8]
        row_str = ""
        empty_count = 0
        for char in row_chars:
            if char == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += char
        if empty_count > 0:
            row_str += str(empty_count)
        rows.append(row_str)
    return "/".join(rows)

