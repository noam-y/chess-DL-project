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

class ResNetTwoHeadWithTriplet(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(ResNetTwoHeadWithTriplet, self).__init__()
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
        out_occ = self.head_occ(features)
        out_piece = self.head_piece(features)
        return out_occ, out_piece, features

class ModelV6Triplet(BaseChessModel):
    def create_model(self) -> nn.Module:
        return ResNetTwoHeadWithTriplet(freeze_backbone=True)

    def fen_to_labels(self, fen_char: str) -> tuple:
        """Returns (Occupancy Tensor, Piece Tensor)"""
        if fen_char == 'e':
            # The piece label for empty squares is a placeholder and will be masked out during loss calculation.
            return (torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long))
        else:
            piece_id = PIECE_TO_ID[fen_char]
            return (torch.tensor(1, dtype=torch.long), torch.tensor(piece_id, dtype=torch.long))
        
    def compute_loss(self, model, batch, device, criterion=None):
        boards, t_occ, t_piece = batch
        t_occ = t_occ.view(-1).to(device)
        t_piece = t_piece.view(-1).to(device)
        inputs = boards.view(-1, 3, 96, 96).to(device)

        out_occ, out_piece, features = model(inputs)
        
        # 1. Occupancy Loss
        criterion_occ = nn.CrossEntropyLoss()
        loss_occ = criterion_occ(out_occ, t_occ)
        
        # 2. Conditional Piece Losses (Only for occupied squares)
        mask = (t_occ == 1)
        loss_piece_ce = torch.tensor(0.0, device=device)
        loss_triplet = torch.tensor(0.0, device=device)
        
        if mask.sum() > 1: # Need at least 2 occupied samples for pairwise distances
            occupied_features = features[mask]
            occupied_t_piece = t_piece[mask]
            occupied_out_piece = out_piece[mask]
            
            # a. Piece Classification Loss (CrossEntropy)
            criterion_piece = nn.CrossEntropyLoss()
            loss_piece_ce = criterion_piece(occupied_out_piece, occupied_t_piece)
            
            # b. Triplet Loss (Batch Hard Mining)
            if len(occupied_t_piece.unique()) > 1: # Need at least two different classes for triplet loss
                pairwise_dist = torch.cdist(occupied_features, occupied_features, p=2)
                
                labels_matrix = occupied_t_piece.unsqueeze(0) == occupied_t_piece.unsqueeze(1)
                labels_matrix.fill_diagonal_(False)
                
                positive_mask = labels_matrix
                negative_mask = ~labels_matrix
                
                # For each anchor, find the hardest positive (max distance)
                hardest_positives = (pairwise_dist * positive_mask.float()).max(dim=1)[0]
                
                # For each anchor, find the hardest negative (min distance)
                max_dist = pairwise_dist.max().item()
                hardest_negatives = (pairwise_dist + (max_dist + 1) * (~negative_mask).float()).min(dim=1)[0]
                
                margin = 1.0
                triplet_loss = F.relu(margin + hardest_positives - hardest_negatives).mean()
                loss_triplet = triplet_loss

        # Combine losses (equal weighting for now)
        total_loss = loss_occ + loss_piece_ce + loss_triplet
        
        # Metrics calculation
        pred_occ = torch.argmax(out_occ, 1)
        pred_piece = torch.argmax(out_piece, 1)
        
        unified_targets = torch.where(t_occ == 0, 0, t_piece + 1)
        unified_preds = torch.where(pred_occ == 0, 0, pred_piece + 1)
        correct = (unified_preds == unified_targets).sum().item()
        
        metrics = {
            'correct': correct, 
            'total': t_occ.size(0),
            'preds': unified_preds.cpu().numpy(),
            'targets': unified_targets.cpu().numpy(),
            'loss_occ': loss_occ.item(),
            'loss_piece_ce': loss_piece_ce.item(),
            'loss_triplet': loss_triplet.item()
        }
        
        return total_loss, metrics

    def get_optimizer(self, model, lr=0.001):
        # Only optimize the parameters of the heads, as the backbone is frozen.
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(params_to_update, lr=lr, weight_decay=1e-4)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            out_occ, out_piece, _ = model(tile_tensor) # Ignore features during inference
            
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
        Unfreeze layers progressively.
        Epochs are 0-indexed.
        """
        # After epoch 2 (i.e. starting epoch 3), unfreeze the last 3 layers of the backbone
        if epoch == 2:
            print("Unfreezing last 3 layers of backbone...")
            # ResNet backbone is a Sequential model.
            # We can access children. The last few children are the residual blocks.
            # Let's unfreeze the last 3 children of the backbone.
            # Note: The backbone in ResNetTwoHeadWithTriplet is defined as:
            # self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])
            
            # Let's see how many children there are.
            children = list(model.backbone.children())
            # Typically ResNet18 has:
            # 0: Conv1
            # 1: BN
            # 2: ReLU
            # 3: MaxPool
            # 4: Layer1 (BasicBlock)
            # 5: Layer2 (BasicBlock)
            # 6: Layer3 (BasicBlock)
            # 7: Layer4 (BasicBlock)
            # Total 8 children in the backbone (since fc and avgpool are removed or avgpool is last?)
            # Actually, `list(self.resnet.children())[:-1]` removes the FC layer.
            # The standard ResNet children are: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc.
            # So `[:-1]` removes `fc`. `avgpool` is at index -1 of the backbone.
            
            # We want to unfreeze the last 3 layers: avgpool, layer4, layer3.
            
            layers_to_unfreeze = children[-3:]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Update optimizer to include these new parameters
            # We need to add the newly unfrozen parameters to the optimizer
            # The easiest way is to re-initialize the optimizer or add param groups.
            # However, `train_all.py` doesn't re-fetch the optimizer.
            # We can manually add the parameters to the optimizer.
            
            new_params = []
            for layer in layers_to_unfreeze:
                new_params.extend([p for p in layer.parameters() if p.requires_grad])
            
            # Filter out params that are already in the optimizer
            existing_params = set()
            for group in optimizer.param_groups:
                for p in group['params']:
                    existing_params.add(p)
            
            params_to_add = [p for p in new_params if p not in existing_params]
            
            if params_to_add:
                optimizer.add_param_group({'params': params_to_add})
                print(f"Added {len(params_to_add)} parameters to optimizer.")

        # After epoch 5 (i.e. starting epoch 6), unfreeze more layers (e.g., layer2, layer1)
        if epoch == 5:
            print("Unfreezing more layers (layer1, layer2)...")
            children = list(model.backbone.children())
            # Unfreeze layer1 and layer2 (indices 4 and 5 typically, or -5 and -4)
            # Let's just unfreeze everything from index 4 onwards (layer1..avgpool)
            # children[-5:-3] would be layer1 and layer2
            
            layers_to_unfreeze = children[-5:-3]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
            
            new_params = []
            for layer in layers_to_unfreeze:
                new_params.extend([p for p in layer.parameters() if p.requires_grad])
            
            existing_params = set()
            for group in optimizer.param_groups:
                for p in group['params']:
                    existing_params.add(p)
            
            params_to_add = [p for p in new_params if p not in existing_params]
            
            if params_to_add:
                optimizer.add_param_group({'params': params_to_add})
                print(f"Added {len(params_to_add)} parameters to optimizer.")
