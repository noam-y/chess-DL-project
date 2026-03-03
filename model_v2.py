import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from chess_model_protocol import Piece
from base_model import BaseChessModel

ID_TO_PIECE = {
    0: 'e', 
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}

class SmartChessNetV2(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNetV2, self).__init__()
        try:
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except:
            self.base_model = models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

class ModelV2(BaseChessModel):
    def create_model(self) -> nn.Module:
        return SmartChessNetV2()

    def fen_to_labels(self, fen: str) -> tuple:
        # For the new dataset, fen is a single character string (e.g., 'P', 'e', 'k')
        # We need to return a single tensor with the class ID.
        piece_to_id = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }
        
        # If fen is 'e' or not in map, it's empty (0)
        label_id = piece_to_id.get(fen, 0)
        return (torch.tensor(label_id, dtype=torch.long),)

    def compute_loss(self, model, batch, device, criterion=None):
        # batch is (images, labels)
        # images: [batch_size, 3, 96, 96]
        # labels: [batch_size]
        images, labels = batch
        inputs = images.to(device)
        targets = labels.to(device)
        
        if criterion is None:
            class_weights = torch.ones(13).to(device)
            class_weights[0] = 0.2 
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        _, preds = torch.max(outputs, 1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        
        return loss, {'correct': correct, 'total': total}

    def get_optimizer(self, model, lr=0.001):
        return optim.Adam(model.parameters(), lr=lr)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        with torch.no_grad():
            output = model(tile_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            if conf.item() < threshold:
                return Piece.OOD
            
            char = ID_TO_PIECE[pred_idx.item()]
            return Piece(char)
