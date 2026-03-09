import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from chess_model_protocol import Piece
from base_model import BaseChessModel

# Mapping for inference
TYPE_ID_TO_CHAR = {1: 'p', 2: 'n', 3: 'b', 4: 'r', 5: 'q', 6: 'k'}

class SmartChessNetV4(nn.Module):
    def __init__(self):
        super(SmartChessNetV4, self).__init__()
        # Architecture from train_kfold.py
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        # Input is 96x96 (from BaseChessDataset)
        # Pool1: 48x48
        # Pool2: 24x24
        # Pool3: 12x12
        self.fc_dim = 128 * 12 * 12
        
        self.head_occ = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        self.head_color = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 3))
        self.head_type = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):
        features = self.backbone(x)
        return self.head_occ(features), self.head_color(features), self.head_type(features)

class ModelV4(BaseChessModel):
    def create_model(self) -> nn.Module:
        return SmartChessNetV4()

    def fen_to_labels(self, fen: str) -> tuple:
        # fen is a single character string (e.g., 'P', 'e', 'k')
        piece_to_id = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6}
        
        if fen == 'e':
            occ = 0
            color = 0 # Empty color
            piece_type = 0 # Empty type
        else:
            occ = 1
            color = 1 if fen.isupper() else 2 # 1=White, 2=Black
            piece_type = piece_to_id[fen.lower()]

        return (torch.tensor(occ, dtype=torch.long), 
                torch.tensor(color, dtype=torch.long), 
                torch.tensor(piece_type, dtype=torch.long))

    def compute_loss(self, model, batch, device, criterion=None):
        # batch is (images, occ, color, type)
        images, t_occ, t_color, t_type = batch
        
        inputs = images.to(device)
        t_occ = t_occ.to(device)
        t_color = t_color.to(device)
        t_type = t_type.to(device)

        p_occ, p_color, p_type = model(inputs)
        
        criterion = nn.CrossEntropyLoss()
        
        loss = criterion(p_occ, t_occ) + criterion(p_color, t_color) + criterion(p_type, t_type)
        
        # Accuracy: All 3 heads must be correct
        pred_occ = torch.argmax(p_occ, 1)
        pred_color = torch.argmax(p_color, 1)
        pred_type = torch.argmax(p_type, 1)
        
        correct = (pred_occ == t_occ) & (pred_color == t_color) & (pred_type == t_type)
        total = t_occ.size(0)
        
        return loss, {'correct': correct.sum().item(), 'total': total}

    def get_optimizer(self, model, lr=0.001):
        return optim.Adam(model.parameters(), lr=lr)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            p_occ, p_color, p_type = model(tile_tensor)
            
            prob_occ = F.softmax(p_occ, dim=1)
            prob_color = F.softmax(p_color, dim=1)
            prob_type = F.softmax(p_type, dim=1)
            
            conf_occ, pred_occ = torch.max(prob_occ, 1)
            conf_color, pred_color = torch.max(prob_color, 1)
            conf_type, pred_type = torch.max(prob_type, 1)
            
            # Check OOD based on occupancy confidence
            if conf_occ.item() < threshold:
                return Piece.OOD
                
            # If predicted empty
            if pred_occ.item() == 0:
                return Piece.EMPTY
            
            # If occupied, check other confidences
            if conf_color.item() < threshold or conf_type.item() < threshold:
                return Piece.OOD
                
            # Construct piece
            type_id = pred_type.item()
            color_id = pred_color.item()
            
            if type_id not in TYPE_ID_TO_CHAR:
                return Piece.OOD # Should not happen if trained correctly, but safety check
                
            char = TYPE_ID_TO_CHAR[type_id]
            
            if color_id == 1: # White
                char = char.upper()
            elif color_id == 2: # Black
                char = char.lower()
            else:
                # Occupied but color is 0 (Empty)? Contradiction.
                return Piece.OOD
                
            return Piece(char)
