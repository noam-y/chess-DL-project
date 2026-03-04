import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score

# Import your dataset logic
from base_model import BaseChessDataset

# --- GLOBAL MAPPINGS ---
CHAR_TO_UNIFIED = {'e': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                   'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
CHAR_TO_PIECE12 = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
CHAR_TO_PIECE6 = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}


# --- DYNAMIC CONFIGURABLE MODEL ---
class ConfigurableChessResNet(nn.Module):
    def __init__(self, num_heads, use_freeze):
        super().__init__()
        self.num_heads = num_heads
        self.use_freeze = use_freeze

        # Using standard ResNet18 load to avoid Hub dependency issues
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if self.use_freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        num_ftrs = resnet.fc.in_features

        if self.num_heads == 1:
            self.head_main = nn.Linear(num_ftrs, 13)
        elif self.num_heads == 2:
            self.head_occ = nn.Linear(num_ftrs, 2)
            self.head_piece = nn.Linear(num_ftrs, 12)
        elif self.num_heads == 3:
            self.head_occ = nn.Linear(num_ftrs, 2)
            self.head_color = nn.Linear(num_ftrs, 2)
            self.head_piece = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)

        if self.num_heads == 1:
            return self.head_main(features), features
        elif self.num_heads == 2:
            return self.head_occ(features), self.head_piece(features), features
        elif self.num_heads == 3:
            return self.head_occ(features), self.head_color(features), self.head_piece(features), features


# --- HELPER FUNCTIONS ---
def get_sampler(sampling_type, labels):
    if sampling_type == 'none' or not labels:
        return None

    class_counts = Counter(labels)
    sample_weights = []

    if sampling_type == 'uniform':
        for label in labels:
            sample_weights.append(1.0 / max(1, class_counts[label]))
    elif sampling_type == '50_50':
        for label in labels:
            if label == 'e':
                sample_weights.append(0.5 / max(1, class_counts['e']))
            else:
                sample_weights.append((0.5 / 12.0) / max(1, class_counts[label]))

    tensor_weights = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(tensor_weights, len(tensor_weights), replacement=True)


def progressive_unfreeze(model, epoch, optimizer):
    if not hasattr(model, 'use_freeze') or not model.use_freeze:
        return
    if epoch == 2:
        for layer in list(model.backbone.children())[-3:]:
            for param in layer.parameters(): param.requires_grad = True
    elif epoch == 5:
        for layer in list(model.backbone.children())[-5:-3]:
            for param in layer.parameters(): param.requires_grad = True

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer.param_groups[0]['params'] = params_to_update


def chars_to_tensors(chars, device):
    t_unified = torch.tensor([CHAR_TO_UNIFIED[c] for c in chars], dtype=torch.long, device=device)
    t_occ = torch.tensor([0 if c == 'e' else 1 for c in chars], dtype=torch.long, device=device)
    t_color = torch.tensor([0 if c == 'e' else (1 if c.isupper() else 0) for c in chars], dtype=torch.long,
                           device=device)
    t_piece12 = torch.tensor([0 if c == 'e' else CHAR_TO_PIECE12[c] for c in chars], dtype=torch.long, device=device)
    t_piece6 = torch.tensor([0 if c == 'e' else CHAR_TO_PIECE6[c.lower()] for c in chars], dtype=torch.long,
                            device=device)
    return t_unified, t_occ, t_color, t_piece12, t_piece6


def calculate_triplet_loss(features, targets, mask, device):
    if mask.sum() < 2: return torch.tensor(0.0, device=device)

    occ_features = features[mask]
    occ_targets = targets[mask]

    if len(occ_targets.unique()) < 2: return torch.tensor(0.0, device=device)

    pairwise_dist = torch.cdist(occ_features, occ_features, p=2)
    labels_matrix = occ_targets.unsqueeze(0) == occ_targets.unsqueeze(1)
    labels_matrix.fill_diagonal_(False)

    positive_mask = labels_matrix
    negative_mask = ~labels_matrix

    hardest_positives = (pairwise_dist * positive_mask.float()).max(dim=1)[0]
    max_dist = pairwise_dist.max().item()
    hardest_negatives = (pairwise_dist + (max_dist + 1) * (~negative_mask).float()).min(dim=1)[0]

    margin = 1.0
    triplet_loss = F.relu(margin + hardest_positives - hardest_negatives).mean()
    return triplet_loss


# --- MAIN BABY GRID SEARCH ---
def main():
    data_dir = "assets/new_dataset"
    output_base = "baby_grid_results"
    os.makedirs(output_base, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting 5-Combination 'Baby' Grid Search on {device}...")

    # 5 Specific Configurations to test every code path
    # Format: (Heads, Sampling, Triplet, Freeze)
    test_combinations = [
        (1, 'none', False, True),  # The Baseline
        (2, '50_50', False, False),  # 2-Head, dynamic sampler, full unfreeze
        (2, 'uniform', True, True),  # 2-Head, Triplet Loss active
        (3, 'none', False, False),  # 3-Head Color separation test
        (3, '50_50', True, True)  # The kitchen sink
    ]

    all_games = ['game1', 'game2', 'game3', 'game4']
    results = []
    combination_idx = 1

    def simple_fen(char):
        return char

    for heads, sampling, use_triplet, freeze in test_combinations:
        config_name = f"H{heads}_S-{sampling}_T-{use_triplet}_F-{freeze}"
        print(f"\n{'=' * 60}")
        print(f"Model {combination_idx}/5: {config_name}")
        print(f"{'=' * 60}")

        config_dir = os.path.join(output_base, config_name)
        os.makedirs(config_dir, exist_ok=True)
        fold_f1_scores = []

        for val_game in all_games:
            print(f"\n--- Fold: Validating on {val_game} ---")

            train_ds = BaseChessDataset(data_dir, mode='train', val_game_name=val_game, fen_converter=simple_fen)
            val_ds = BaseChessDataset(data_dir, mode='val', val_game_name=val_game, fen_converter=simple_fen)

            if len(train_ds) == 0: continue

            sampler = get_sampler(sampling, train_ds.all_labels)
            train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, shuffle=(sampler is None),
                                      num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

            model = ConfigurableChessResNet(heads, freeze).to(device)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)

            best_fold_f1 = 0.0

            # BABY TEST: Only 3 epochs instead of 100
            for epoch in range(3):
                progressive_unfreeze(model, epoch, optimizer)

                # --- TRAIN LOOP ---
                model.train()
                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train", leave=False):
                    if batch is None: continue
                    boards, chars = batch
                    inputs = boards.to(device)

                    t_unified, t_occ, t_color, t_piece12, t_piece6 = chars_to_tensors(chars, device)

                    optimizer.zero_grad()
                    total_loss = torch.tensor(0.0, device=device)
                    mask = (t_occ == 1)

                    if heads == 1:
                        out_main, features = model(inputs)
                        total_loss += nn.CrossEntropyLoss()(out_main, t_unified)
                        if use_triplet:
                            total_loss += calculate_triplet_loss(features, t_unified, mask, device)

                    elif heads == 2:
                        out_occ, out_piece, features = model(inputs)
                        total_loss += nn.CrossEntropyLoss()(out_occ, t_occ)
                        if mask.sum() > 0:
                            total_loss += nn.CrossEntropyLoss()(out_piece[mask], t_piece12[mask])
                        if use_triplet:
                            total_loss += calculate_triplet_loss(features, t_piece12, mask, device)

                    elif heads == 3:
                        out_occ, out_color, out_piece, features = model(inputs)
                        total_loss += nn.CrossEntropyLoss()(out_occ, t_occ)
                        if mask.sum() > 0:
                            total_loss += nn.CrossEntropyLoss()(out_color[mask], t_color[mask])
                            total_loss += nn.CrossEntropyLoss()(out_piece[mask], t_piece6[mask])
                        if use_triplet:
                            total_loss += calculate_triplet_loss(features, t_piece12, mask, device)

                    total_loss.backward()
                    optimizer.step()

                # --- VALIDATION LOOP ---
                model.eval()
                all_preds, all_targets = [], []

                with torch.no_grad():
                    for batch in val_loader:
                        if batch is None: continue
                        boards, chars = batch
                        inputs = boards.to(device)
                        t_unified, t_occ, t_color, t_piece12, t_piece6 = chars_to_tensors(chars, device)

                        if heads == 1:
                            out_main, _ = model(inputs)
                            preds = torch.argmax(out_main, 1)
                        elif heads == 2:
                            out_occ, out_piece, _ = model(inputs)
                            p_occ = torch.argmax(out_occ, 1)
                            p_piece = torch.argmax(out_piece, 1)
                            preds = torch.where(p_occ == 0, 0, p_piece + 1)
                        elif heads == 3:
                            out_occ, out_color, out_piece, _ = model(inputs)
                            p_occ = torch.argmax(out_occ, 1)
                            p_color = torch.argmax(out_color, 1)
                            p_piece6 = torch.argmax(out_piece, 1)
                            reconstructed_piece = torch.where(p_color == 1, p_piece6 + 1, p_piece6 + 7)
                            preds = torch.where(p_occ == 0, 0, reconstructed_piece)

                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(t_unified.cpu().numpy())

                val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100

                if val_f1 > best_fold_f1:
                    best_fold_f1 = val_f1
                    torch.save(model.state_dict(), os.path.join(config_dir, f"best_{val_game}.pth"))

            fold_f1_scores.append(best_fold_f1)

        mean_cv_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0
        print(f"\n>>> Configuration [{config_name}] Final Mean 4-Fold F1: {mean_cv_f1:.2f}%")

        results.append({
            "Config ID": config_name,
            "Heads": heads,
            "Sampling": sampling,
            "Triplet Loss": use_triplet,
            "Freeze Backbone": freeze,
            "Mean 4-Fold F1": mean_cv_f1
        })

        pd.DataFrame(results).to_csv(os.path.join(output_base, "baby_running_results.csv"), index=False)
        combination_idx += 1

    print("\n=============================================")
    print("BABY GRID SEARCH COMPLETE!")
    print("=============================================")
    final_df = pd.DataFrame(results).sort_values(by="Mean 4-Fold F1", ascending=False)
    final_df.to_csv(os.path.join(output_base, "BABY_FINAL_RESULTS.csv"), index=False)
    print(final_df.to_markdown(index=False))


if __name__ == "__main__":
    main()