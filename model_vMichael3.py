import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

from chess_model_protocol import Piece
from base_model import BaseChessModel

# Dictionary mapping for 13 classes
ID_TO_PIECE = {
    0: 'e', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}
PIECE_TO_ID = {v: k for k, v in ID_TO_PIECE.items()}

class ResNetUnified13(nn.Module):
    def __init__(self):
        super(ResNetUnified13, self).__init__()
        
        # Load pre-trained ResNet18
        # NO FREEZING! We want the whole network to learn the specific wood textures.
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer to output exactly 13 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 13)

    def forward(self, x):
        return self.resnet(x)


class ModelUnified(BaseChessModel):
    def create_model(self) -> nn.Module:
        return ResNetUnified13()

    def fen_to_labels(self, fen_char: str) -> tuple:
        """Converts a SINGLE character to a single target tensor ID"""
        class_id = PIECE_TO_ID.get(fen_char, 0) # Defaults to 0 ('e') if unknown
        return (torch.tensor(class_id, dtype=torch.long),)
        
    def compute_loss(self, model, batch, device, criterion=None):
        # Batch now only contains the image and a single 13-class label
        boards, labels_tuple = batch
        
        # Extract the single tensor from the tuple returned by the dataset
        labels = labels_tuple[0] 
        
        inputs = boards.view(-1, 3, 96, 96).to(device)
        targets = labels.view(-1).to(device)

        outputs = model(inputs)
        
        # Uniform weights - the Sampler in train_all handles the imbalance!
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        loss = criterion(outputs, targets)
        
        # Metrics Calculation
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        
        return loss, {'correct': correct, 'total': targets.size(0)}

    def get_optimizer(self, model, lr=0.001):
        # Update ALL parameters since we didn't freeze anything
        return optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tile_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
            class_id = pred.item()
            char = ID_TO_PIECE[class_id]
            
            # Simple OOD Logic: If confidence is below threshold, return OOD
            if conf.item() < threshold:
                return Piece.OOD
            
            if char == 'e':
                return Piece.EMPTY
            
            return Piece(char)