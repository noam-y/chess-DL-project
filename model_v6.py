import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F

from chess_model_protocol import Piece
from base_model import BaseChessModel

# Dictionary mapping for ONLY the 12 active pieces
PIECE_TO_ID = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

class ResNetTwoHead(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(ResNetTwoHead, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        
        # Head 1: Occupancy (Empty=0, Occupied=1)
        self.head_occ = nn.Linear(num_ftrs, 2)
        # Head 2: Piece Identity (12 classes)
        self.head_piece = nn.Linear(num_ftrs, 12)

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        return self.head_occ(features), self.head_piece(features)

class ModelV6(BaseChessModel):
    def create_model(self) -> nn.Module:
        # Start with the backbone frozen
        return ResNetTwoHead(freeze_backbone=True)

    def fen_to_labels(self, fen_char: str) -> tuple:
        """Returns (Occupancy Tensor, Piece Tensor)"""
        if fen_char == 'e':
            return (torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long))
        else:
            piece_id = PIECE_TO_ID[fen_char]
            return (torch.tensor(1, dtype=torch.long), torch.tensor(piece_id, dtype=torch.long))
        
    def compute_loss(self, model, batch, device, criterion=None):
        boards, t_occ, t_piece = batch
        t_occ = t_occ.view(-1).to(device)
        t_piece = t_piece.view(-1).to(device)
        inputs = boards.view(-1, 3, 96, 96).to(device)

        out_occ, out_piece = model(inputs)
        
        # 1. Base Occupancy Loss
        criterion_occ = nn.CrossEntropyLoss()
        loss_occ = criterion_occ(out_occ, t_occ)
        
        # 2. Conditional Piece Loss (Only calculate if square is occupied)
        mask = (t_occ == 1)
        loss_piece = torch.tensor(0.0, device=device)
        if mask.sum() > 0:
            criterion_piece = nn.CrossEntropyLoss()
            loss_piece = criterion_piece(out_piece[mask], t_piece[mask])
            
        total_loss = loss_occ + loss_piece
        
        # 3. Create Unified Arrays for external F1-Score evaluation
        pred_occ = torch.argmax(out_occ, 1)
        pred_piece = torch.argmax(out_piece, 1)
        
        # Map back to a 0-12 unified space (0=Empty, 1-12=Pieces) for the metric report
        unified_targets = torch.where(t_occ == 0, 0, t_piece + 1)
        unified_preds = torch.where(pred_occ == 0, 0, pred_piece + 1)
        correct = (unified_preds == unified_targets).sum().item()
        
        metrics = {
            'correct': correct, 
            'total': t_occ.size(0),
            'preds': unified_preds.cpu().numpy(),
            'targets': unified_targets.cpu().numpy()
        }
        
        return total_loss, metrics

    def get_optimizer(self, model, lr=0.001):
        # Initially, only the heads have requires_grad=True
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(params_to_update, lr=lr, weight_decay=1e-4)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            out_occ, out_piece = model(tile_tensor)
            
            # 1. Check Occupancy First
            prob_occ = F.softmax(out_occ, dim=1)
            conf_occ, p_occ = torch.max(prob_occ, 1)
            
            if p_occ.item() == 0:
                if conf_occ.item() < threshold:
                    return Piece.OOD
                return Piece.EMPTY
                
            # 2. If Occupied, determine the piece
            prob_piece = F.softmax(out_piece, dim=1)
            conf_piece, p_piece = torch.max(prob_piece, 1)
            
            if conf_occ.item() < threshold or conf_piece.item() < threshold:
                return Piece.OOD
            
            char = ID_TO_PIECE[p_piece.item()]
            return Piece(char)

    def on_epoch_end(self, model, epoch, optimizer):
        """
        Unfreeze layers progressively based on the epoch number.
        Epochs are 0-indexed.
        """
        # The backbone has 9 children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
        children = list(model.backbone.children())
        
        layers_to_unfreeze = []
        
        # After 10 epochs (at the end of epoch index 9)
        if epoch == 9:
            print("\n[Epoch 10] Unfreezing last 2 layers of backbone (layer4, avgpool)...")
            layers_to_unfreeze = children[-2:]
            
        # After 20 epochs (at the end of epoch index 19)
        elif epoch == 19:
            print("\n[Epoch 20] Unfreezing 2 more layers (layer2, layer3)...")
            layers_to_unfreeze = children[-4:-2]

        # After 30 epochs (at the end of epoch index 29)
        elif epoch == 29:
            print("\n[Epoch 30] Unfreezing all remaining backbone layers...")
            layers_to_unfreeze = children[:-4]

        if not layers_to_unfreeze:
            return

        # Unfreeze and collect new parameters
        new_params = []
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    new_params.append(param)
        
        # Add newly unfrozen parameters to the optimizer
        if new_params:
            optimizer.add_param_group({'params': new_params})
            print(f"Added {len(new_params)} parameters to the optimizer.")
