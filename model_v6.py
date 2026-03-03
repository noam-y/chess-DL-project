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

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # Prevent log(0) by adding epsilon
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        # Avoid division by zero if there are no positives for an anchor
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Only average over anchors that actually had positives
        valid_anchors = (mask.sum(1) > 0).float()
        if valid_anchors.sum() > 0:
            loss = (loss * valid_anchors).sum() / valid_anchors.sum()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss

class ResNetTwoHeadWithSupCon(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(ResNetTwoHeadWithSupCon, self).__init__()
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
        
        # Projection Head for SupCon
        self.head_proj = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, 128)
        )

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        out_occ = self.head_occ(features)
        out_piece = self.head_piece(features)
        out_proj = self.head_proj(features)
        
        # Normalize projection features for cosine similarity
        out_proj = F.normalize(out_proj, dim=1)
        
        return out_occ, out_piece, out_proj

class ModelV6(BaseChessModel):
    def create_model(self) -> nn.Module:
        # Start with the backbone frozen
        return ResNetTwoHeadWithSupCon(freeze_backbone=True)

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

        out_occ, out_piece, out_proj = model(inputs)
        
        # 1. Base Occupancy Loss
        criterion_occ = nn.CrossEntropyLoss()
        loss_occ = criterion_occ(out_occ, t_occ)
        
        # 2. Conditional Piece Loss & SupCon Loss (Only calculate if square is occupied)
        mask = (t_occ == 1)
        loss_piece = torch.tensor(0.0, device=device)
        loss_supcon = torch.tensor(0.0, device=device)
        
        if mask.sum() > 1: # Need at least 2 samples for contrastive loss
            # Cross Entropy for Piece Classification
            criterion_piece = nn.CrossEntropyLoss()
            loss_piece = criterion_piece(out_piece[mask], t_piece[mask])
            
            # SupCon Loss
            features_supcon = out_proj[mask].unsqueeze(1)
            labels_supcon = t_piece[mask]
            
            # Only compute SupCon if we have at least 2 classes in the batch
            # AND if there is at least one class with more than 1 sample
            unique_labels, counts = torch.unique(labels_supcon, return_counts=True)
            if len(unique_labels) > 1 and (counts > 1).any():
                criterion_supcon = SupConLoss(temperature=0.1)
                loss_supcon = criterion_supcon(features_supcon, labels_supcon)
            
        total_loss = loss_occ + loss_piece + loss_supcon
        
        # 3. Create Unified Arrays for external F1-Score evaluation
        pred_occ = torch.argmax(out_occ, 1)
        pred_piece = torch.argmax(out_piece, 1)
        
        # Map back to 0-12 unified space
        unified_targets = torch.where(t_occ == 0, 0, t_piece + 1)
        unified_preds = torch.where(pred_occ == 0, 0, pred_piece + 1)
        correct = (unified_preds == unified_targets).sum().item()
        
        metrics = {
            'correct': correct, 
            'total': t_occ.size(0),
            'preds': unified_preds.cpu().numpy(),
            'targets': unified_targets.cpu().numpy(),
            'loss_occ': loss_occ.item(),
            'loss_piece': loss_piece.item(),
            'loss_supcon': loss_supcon.item()
        }
        
        return total_loss, metrics

    def get_optimizer(self, model, lr=0.001):
        # Initially, only the heads have requires_grad=True
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(params_to_update, lr=lr, weight_decay=1e-4)

    def infer_tile(self, model, tile_tensor, device, threshold=0.7) -> Piece:
        tile_tensor = tile_tensor.to(device).unsqueeze(0)
        
        with torch.no_grad():
            out_occ, out_piece, _ = model(tile_tensor)
            
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
