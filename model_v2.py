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
        piece_to_id = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }
        board_tensor = np.zeros((8, 8), dtype=np.int64)
        board_state = fen.split(' ')[0]
        rows = board_state.split('/')
        for r, row_str in enumerate(rows):
            c = 0
            for char in row_str:
                if char.isdigit():
                    c += int(char)
                else:
                    if char in piece_to_id:
                        board_tensor[r, c] = piece_to_id[char]
                    c += 1
        return (torch.from_numpy(board_tensor),)

    def compute_loss(self, model, batch, device, criterion=None):
        # batch is (patches, labels)
        boards, labels = batch
        inputs = boards.view(-1, 3, 96, 96).to(device)
        targets = labels.view(-1).to(device)
        
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
