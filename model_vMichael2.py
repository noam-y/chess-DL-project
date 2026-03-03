import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

# Import from your existing protocol files
from chess_model_protocol import Piece
from base_model import BaseChessModel

PIECE_TYPES_INV = {0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k'}

class ResNetMultiHeadV4(nn.Module):
    def __init__(self):
        super(ResNetMultiHeadV4, self).__init__()
        
        # Load pre-trained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 1. FREEZING STRATEGY: 
        # Freeze the early layers (conv1, bn1, layer1, layer2) because they hold 
        # fundamental edge/color detectors from ImageNet.
        # We leave layer3 and layer4 unfrozen so they can learn complex chess shapes.
        layers_to_freeze = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']
        for name, child in resnet.named_children():
            if name in layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False
                    
        # Extract the backbone (everything except the final fully connected layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features # Will be 512
        
        # 2. THE THREE HEADS:
        self.head_occ = nn.Linear(num_ftrs, 2)   # 0: Empty, 1: Occupied
        self.head_color = nn.Linear(num_ftrs, 2) # 0: Black, 1: White
        self.head_piece = nn.Linear(num_ftrs, 6) # 6 Piece types

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1) # Flatten the 1x1 spatial grid
        
        out_occ = self.head_occ(features)
        out_color = self.head_color(features)
        out_piece = self.head_piece(features)
        
        return out_occ, out_color, out_piece


class ModelV4(BaseChessModel):
    def create_model(self) -> nn.Module:
        return ResNetMultiHeadV4()

    def fen_to_labels(self, fen_char: str) -> tuple:
        """Converts a SINGLE character (e.g., 'Q', 'p', 'e') to 3 target tensors"""
        type_map = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
        
        if fen_char == 'e':
            occ = torch.tensor(0, dtype=torch.long)
            color = torch.tensor(0, dtype=torch.long) # Dummy value, ignored by loss mask
            piece = torch.tensor(0, dtype=torch.long) # Dummy value, ignored by loss mask
        else:
            occ = torch.tensor(1, dtype=torch.long)
            color = torch.tensor(1 if fen_char.isupper() else 0, dtype=torch.long)
            piece = torch.tensor(type_map[fen_char.lower()], dtype=torch.long)
            
        return (occ, color, piece)
        
    def compute_loss(self, model, batch, device, criterion=None):
        boards, l_occ, l_color, l_piece = batch
        
        inputs = boards.view(-1, 3, 96, 96).to(device)
        t_occ = l_occ.view(-1).to(device)
        t_color = l_color.view(-1).to(device)
        t_piece = l_piece.view(-1).to(device)

        out_occ, out_color, out_piece = model(inputs)
        
        # 3. ADVANCED LOSS WEIGHTING STRATEGY
        # Address class imbalance: Empty squares (0) get very low weight, 
        # Occupied squares (1) get very high weight.
        weight_occ = torch.tensor([0.2, 2.0], dtype=torch.float32).to(device) 
        
        # Color is usually 50/50, standard weights
        weight_color = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
        
        # Heavily punish piece misclassification (Acting as the Triplet Loss alternative)
        # Scaled up significantly so the model prioritizes fixing piece errors
        weight_piece = torch.tensor([2.0, 3.0, 3.0, 2.0, 5.0, 4.0], dtype=torch.float32).to(device)

        criterion_occ = nn.CrossEntropyLoss(weight=weight_occ)
        criterion_color = nn.CrossEntropyLoss(weight=weight_color)
        criterion_piece = nn.CrossEntropyLoss(weight=weight_piece)
        
        # Calculate base occupancy loss
        loss_occ = criterion_occ(out_occ, t_occ)
        
        # ONLY calculate color and piece loss on squares that are ACTUALLY occupied
        mask = (t_occ == 1)
        
        # We multiply the piece loss by a scalar (e.g., 2.0) to further emphasize it
        loss_color = criterion_color(out_color[mask], t_color[mask]) if mask.sum() > 0 else 0
        loss_piece = (criterion_piece(out_piece[mask], t_piece[mask]) * 2.0) if mask.sum() > 0 else 0 
        
        total_loss = loss_occ + loss_color + loss_piece
        
        # Metrics Calculation
        pred_occ = torch.argmax(out_occ, 1)
        pred_color = torch.argmax(out_color, 1)
        pred_piece = torch.argmax(out_piece, 1)
        
        correct_empty = (pred_occ == 0) & (t_occ == 0)
        correct_occupied = (pred_occ == 1) & (t_occ == 1) & (pred_color == t_color) & (pred_piece == t_piece)
        correct_squares = correct_empty | correct_occupied
        
        return total_loss, {'correct': correct_squares.sum().item(), 'total': t_occ.size(0)}

    def get_optimizer(self, model, lr=0.001):
        # We use a slightly smaller learning rate because we are fine-tuning a pre-trained model
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        """Inference routing returning the expected Enum"""
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            out_occ, out_color, out_piece = model(tile_tensor)
            
            prob_occ = F.softmax(out_occ, dim=1)
            prob_color = F.softmax(out_color, dim=1)
            prob_piece = F.softmax(out_piece, dim=1)
            
            conf_occ, p_occ = torch.max(prob_occ, 1)
            _, p_color = torch.max(prob_color, 1)
            conf_piece, p_piece = torch.max(prob_piece, 1)
            
            # Handle Empty and OOD
            if p_occ.item() == 0:
                if conf_occ.item() < threshold:
                    return Piece.OOD
                return Piece.EMPTY
                
            # Handle Pieces
            char = PIECE_TYPES_INV[p_piece.item()]
            if p_color.item() == 1:
                char = char.upper()
                
            # If the model thinks a piece is there, but isn't confident in the occupancy OR the piece type
            if (conf_occ.item() < threshold) or (conf_piece.item() < threshold):
                return Piece.OOD
            
            return Piece(char)