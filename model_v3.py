import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from chess_model_protocol import Piece
from base_model import BaseChessModel

PIECE_TYPES_INV = {0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k'}

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

class ModelV3(BaseChessModel):
    def create_model(self) -> nn.Module:
        return SmartChessNetV3()

    def fen_to_labels(self, fen: str) -> tuple:
        # fen is a single character string (e.g., 'P', 'e', 'k')
        type_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
        
        if fen == 'e':
            occ = 0
            color = 0 # Dummy
            piece = 0 # Dummy
        else:
            occ = 1
            color = 1 if fen.isupper() else 0
            piece = type_map[fen.lower()]
            
        return (torch.tensor(occ, dtype=torch.long), 
                torch.tensor(color, dtype=torch.long), 
                torch.tensor(piece, dtype=torch.long))

    def compute_loss(self, model, batch, device, criterion=None):
        # batch is (images, occ, color, piece)
        # images: [batch_size, 3, 96, 96]
        # labels: [batch_size]
        images, t_occ, t_color, t_piece = batch
        
        inputs = images.to(device)
        t_occ = t_occ.to(device)
        t_color = t_color.to(device)
        t_piece = t_piece.to(device)

        out_occ, out_color, out_piece = model(inputs)
        
        weight_occ = torch.tensor([0.722, 1.624], dtype=torch.float32).to(device)
        weight_color = torch.tensor([1.026, 0.976], dtype=torch.float32).to(device)
        weight_piece = torch.tensor([0.319, 1.898, 1.676, 1.222, 3.133, 1.642], dtype=torch.float32).to(device)

        criterion_occ = nn.CrossEntropyLoss(weight=weight_occ)
        criterion_color = nn.CrossEntropyLoss(weight=weight_color)
        criterion_piece = nn.CrossEntropyLoss(weight=weight_piece)
        
        loss_occ = criterion_occ(out_occ, t_occ)
        
        mask = (t_occ == 1)
        loss_color = criterion_color(out_color[mask], t_color[mask]) if mask.sum() > 0 else 0
        loss_piece = criterion_piece(out_piece[mask], t_piece[mask]) if mask.sum() > 0 else 0
        
        loss = loss_occ + loss_color + loss_piece
        
        # Calculate accuracy
        pred_occ = torch.argmax(out_occ, 1)
        pred_color = torch.argmax(out_color, 1)
        pred_piece = torch.argmax(out_piece, 1)
        
        correct_empty = (pred_occ == 0) & (t_occ == 0)
        correct_occupied = (pred_occ == 1) & (t_occ == 1) & (pred_color == t_color) & (pred_piece == t_piece)
        
        correct_squares = correct_empty | correct_occupied
        
        return loss, {'correct': correct_squares.sum().item(), 'total': t_occ.size(0)}

    def get_optimizer(self, model, lr=0.001):
        return optim.Adam(model.parameters(), lr=lr)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            out_occ, out_color, out_piece = model(tile_tensor)
            
            prob_occ = F.softmax(out_occ, dim=1)
            prob_color = F.softmax(out_color, dim=1)
            prob_piece = F.softmax(out_piece, dim=1)
            
            conf_occ, p_occ = torch.max(prob_occ, 1)
            conf_color, p_color = torch.max(prob_color, 1)
            conf_piece, p_piece = torch.max(prob_piece, 1)
            
            if p_occ.item() == 0:
                if conf_occ.item() < threshold:
                    return Piece.OOD
                return Piece.EMPTY
                
            char = PIECE_TYPES_INV[p_piece.item()]
            if p_color.item() == 1:
                char = char.upper()
                
            if (conf_occ.item() < threshold) or (conf_piece.item() < threshold):
                return Piece.OOD
            
            return Piece(char)
