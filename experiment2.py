import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, default_collate, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from collections import Counter
from sklearn.metrics import f1_score
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# --- GLOBAL MAPPINGS ---
CHAR_TO_UNIFIED = {'e': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                   'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
CHAR_TO_PIECE12 = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
CHAR_TO_PIECE6 = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}


# --- CUSTOM ROBUST DATASET ---
class GridDataset(Dataset):
    def __init__(self, data_dir, mode='train', val_game=None, test_game='game5'):
        self.data_dir = data_dir
        self.samples = []
        self.all_labels = []

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        games = [d for d in os.listdir(data_dir) if d.startswith('game') and os.path.isdir(os.path.join(data_dir, d))]

        for game in games:
            if game == test_game and mode != 'test': continue
            if mode == 'test' and game != test_game: continue
            if mode == 'train' and game == val_game: continue
            if mode == 'val' and game != val_game: continue

            csv_path = os.path.join(data_dir, game, 'gt.csv')
            if not os.path.exists(csv_path): continue

            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = os.path.join(data_dir, game, 'tagged_images', row['file_name'])
                char = row['fen']
                self.samples.append((img_path, char))
                self.all_labels.append(char)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, char = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, char
        except Exception:
            return None


# --- CUSTOM COLLATOR ---
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    return default_collate(batch)


# --- DYNAMIC CONFIGURABLE MODEL ---
class ConfigurableChessResNet(nn.Module):
    def __init__(self, num_heads, use_freeze):
        super().__init__()
        self.num_heads = num_heads
        self.use_freeze = use_freeze

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if self.use_freeze:
            for param in self.backbone.parameters(): param.requires_grad = False

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


# --- EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0


# --- HELPER FUNCTIONS ---
def get_sampler(sampling_type, labels):
    if sampling_type == 'none' or not labels: return None
    class_counts = Counter(labels)
    sample_weights = []
    if sampling_type == 'uniform':
        for label in labels: sample_weights.append(1.0 / max(1, class_counts[label]))
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

        # Unfreeze one stage every 10 epochs, from deepest to earliest.
        # backbone children for resnet18 sequential:
        # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool]
    schedule = {
        10: ['layer4'],
        20: ['layer3'],
        30: ['layer2'],
        40: ['layer1'],
        50: ['conv1', 'bn1'],  # relu/maxpool have no trainable parameters
    }
    if epoch not in schedule:
        return

    named_backbone_layers = dict(model.backbone.named_children())
    for layer_name in schedule[epoch]:
        layer = named_backbone_layers.get(layer_name, None)
        if layer is None:
            continue
        for param in layer.parameters():
            param.requires_grad = True

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
    occ_features, occ_targets = features[mask], targets[mask]
    if len(occ_targets.unique()) < 2: return torch.tensor(0.0, device=device)

    pairwise_dist = torch.cdist(occ_features, occ_features, p=2)
    labels_matrix = occ_targets.unsqueeze(0) == occ_targets.unsqueeze(1)
    labels_matrix.fill_diagonal_(False)

    hardest_positives = (pairwise_dist * labels_matrix.float()).max(dim=1)[0]
    hardest_negatives = (pairwise_dist + (pairwise_dist.max().item() + 1) * (~labels_matrix).float()).min(dim=1)[0]
    return F.relu(1.0 + hardest_positives - hardest_negatives).mean()


def calculate_multi_similarity_loss(features, targets, mask, device, alpha=2.0, beta=50.0, base=0.5):
    if mask.sum() < 2:
        return torch.tensor(0.0, device=device)

    occ_features, occ_targets = features[mask], targets[mask]
    if len(occ_targets.unique()) < 2:
        return torch.tensor(0.0, device=device)

    occ_features = F.normalize(occ_features, p=2, dim=1)
    similarity = torch.matmul(occ_features, occ_features.t())
    labels_matrix = occ_targets.unsqueeze(0) == occ_targets.unsqueeze(1)
    eye = torch.eye(labels_matrix.size(0), dtype=torch.bool, device=device)
    pos_mask = labels_matrix & ~eye
    neg_mask = ~labels_matrix

    loss_terms = []
    for i in range(similarity.size(0)):
        pos_sim = similarity[i][pos_mask[i]]
        neg_sim = similarity[i][neg_mask[i]]
        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            continue

        pos_term = (1.0 / alpha) * torch.log1p(torch.exp(-alpha * (pos_sim - base)).sum())
        neg_term = (1.0 / beta) * torch.log1p(torch.exp(beta * (neg_sim - base)).sum())
        loss_terms.append(pos_term + neg_term)

    if not loss_terms:
        return torch.tensor(0.0, device=device)
    return torch.stack(loss_terms).mean()


def freeze_batchnorm_for_frozen_layers(model):
    if not hasattr(model, "backbone"):
        return

    for layer in model.backbone.children():
        layer_params = list(layer.parameters())
        if not layer_params:
            continue
        layer_is_frozen = all(not p.requires_grad for p in layer_params)
        if layer_is_frozen:
            for module in layer.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    if module.weight is not None:
                        module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False


def build_optimizer(model, base_lr, freeze, backbone_lr_scale=0.1):
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    if freeze:
        for p in backbone_params:
            p.requires_grad = False

    return optim.Adam(
        [
            {"params": head_params, "lr": base_lr},
            {"params": backbone_params, "lr": base_lr * backbone_lr_scale},
        ],
        weight_decay=1e-4
    )


# --- ENSEMBLE EVALUATOR ---
def evaluate_ensemble_on_test(config_dir, test_game_name, heads, freeze, device):
    data_dir = "assets/new_dataset"
    test_ds = GridDataset(data_dir, mode='test', test_game=test_game_name)
    if len(test_ds) == 0: return 0.0
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

    models_list = []
    for file in os.listdir(config_dir):
        if file.endswith(".pth"):
            model = ConfigurableChessResNet(heads, freeze).to(device)
            model.load_state_dict(torch.load(os.path.join(config_dir, file), map_location=device))
            model.eval()
            models_list.append(model)

    if not models_list: return 0.0

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            boards, chars = batch
            inputs = boards.to(device)
            t_unified, _, _, _, _ = chars_to_tensors(chars, device)

            ensemble_probs = torch.zeros((inputs.size(0), 13), device=device)
            for model in models_list:
                if heads == 1:
                    out_main, _ = model(inputs)
                    ensemble_probs += F.softmax(out_main, dim=1)
                elif heads == 2:
                    out_occ, out_piece, _ = model(inputs)
                    prob_occ, prob_piece = F.softmax(out_occ, dim=1), F.softmax(out_piece, dim=1)
                    unified_prob = torch.zeros((inputs.size(0), 13), device=device)
                    unified_prob[:, 0] = prob_occ[:, 0]
                    for i in range(12): unified_prob[:, i + 1] = prob_occ[:, 1] * prob_piece[:, i]
                    ensemble_probs += unified_prob
                elif heads == 3:
                    out_occ, out_color, out_piece, _ = model(inputs)
                    prob_occ, prob_color, prob_piece6 = F.softmax(out_occ, dim=1), F.softmax(out_color,
                                                                                             dim=1), F.softmax(
                        out_piece, dim=1)
                    unified_prob = torch.zeros((inputs.size(0), 13), device=device)
                    unified_prob[:, 0] = prob_occ[:, 0]
                    for i in range(6):
                        unified_prob[:, i + 1] = prob_occ[:, 1] * prob_color[:, 1] * prob_piece6[:, i]
                        unified_prob[:, i + 7] = prob_occ[:, 1] * prob_color[:, 0] * prob_piece6[:, i]
                    ensemble_probs += unified_prob

            ensemble_probs /= len(models_list)
            all_preds.extend(torch.argmax(ensemble_probs, dim=1).cpu().numpy())
            all_targets.extend(t_unified.cpu().numpy())

    return f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100


# --- MAIN GRID SEARCH ---
def main():
    data_dir = "assets/new_dataset"
    output_base = "experiment2_results"
    os.makedirs(output_base, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting 72-Combination Grid Search on {device}...")

    heads_opts = [1, 2, 3]
    sample_opts = ['50_50']
    triplet_opts = ['new']
    freeze_opts = [True, False]
    batch_size_opts = [16, 32, 64]

    combs = [[1, '50_50', 'old', True, 32],  # og number 1 seed
             [1, '50_50', 'new', True, 16],  # my improvment
             [3, '50_50', 'old', False, 32], # og number 2 seed
             [3, '50_50', 'new', True, 16],  # my improvment
             [1, 'none', 'old', False, 32],  # og number 3 seed
             [2, '50_50', 'new', True, 16]]  # my guess

    all_games = ['game2', 'game4', 'game6', 'game7']
    results = []

    for combination_idx, (heads, sampling, triplet_mode, freeze, batch_size) in combs:
            # enumerate(itertools.product(heads_opts, sample_opts, triplet_opts, freeze_opts, batch_size_opts), 1)):
        config_name = f"H{heads}_S-{sampling}_T-{triplet_mode}_F-{freeze}_B-{batch_size}"
        print(f"\n{'=' * 60}\nModel {combination_idx}/72: {config_name}\n{'=' * 60}")

        config_dir = os.path.join(output_base, config_name)
        os.makedirs(config_dir, exist_ok=True)
        fold_f1_scores = []

        for val_game in all_games:
            print(f"\n--- Fold: Validating on {val_game} ---")
            train_ds = GridDataset(data_dir, mode='train', val_game=val_game)
            val_ds = GridDataset(data_dir, mode='val', val_game=val_game)
            if len(train_ds) == 0:
                print(f"Skipping {val_game} (No training data found)")
                continue

            sampler = get_sampler(sampling, train_ds.all_labels)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None),
                                      num_workers=4, collate_fn=custom_collate)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

            model = ConfigurableChessResNet(heads, freeze).to(device)
            optimizer = build_optimizer(model, base_lr=0.0001, freeze=freeze, backbone_lr_scale=0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
            early_stopping = EarlyStopping(patience=30)
            ce_criterion = nn.CrossEntropyLoss()

            best_fold_f1 = 0.0

            # MAIN EVENT: Up to 100 epochs
            for epoch in range(70):
                progressive_unfreeze(model, epoch, optimizer)
                model.train()
                freeze_batchnorm_for_frozen_layers(model)
                epoch_ce_loss_sum = 0.0
                epoch_metric_loss_sum = 0.0
                epoch_batches = 0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train", leave=False):
                    if batch is None: continue
                    boards, chars = batch
                    inputs = boards.to(device)
                    t_unified, t_occ, t_color, t_piece12, t_piece6 = chars_to_tensors(chars, device)

                    optimizer.zero_grad()
                    ce_loss = torch.tensor(0.0, device=device)
                    metric_loss = torch.tensor(0.0, device=device)
                    mask = (t_occ == 1)

                    if heads == 1:
                        out_main, features = model(inputs)
                        ce_loss += ce_criterion(out_main, t_unified)
                        if triplet_mode == 'old':
                            metric_loss += calculate_triplet_loss(features, t_unified, mask, device)
                        else:
                            metric_loss += calculate_multi_similarity_loss(features, t_unified, mask, device)
                    elif heads == 2:
                        out_occ, out_piece, features = model(inputs)
                        ce_loss += ce_criterion(out_occ, t_occ)
                        if mask.sum() > 0: ce_loss += ce_criterion(out_piece[mask], t_piece12[mask])
                        if triplet_mode == 'old':
                            metric_loss += calculate_triplet_loss(features, t_piece12, mask, device)
                        else:
                            metric_loss += calculate_multi_similarity_loss(features, t_piece12, mask, device)
                    elif heads == 3:
                        out_occ, out_color, out_piece, features = model(inputs)
                        ce_loss += ce_criterion(out_occ, t_occ)
                        if mask.sum() > 0:
                            ce_loss += ce_criterion(out_color[mask], t_color[mask])
                            ce_loss += ce_criterion(out_piece[mask], t_piece6[mask])
                        if triplet_mode == 'old':
                            metric_loss += calculate_triplet_loss(features, t_piece12, mask, device)
                        else:
                            metric_loss += calculate_multi_similarity_loss(features, t_piece12, mask, device)

                    total_loss = ce_loss + metric_loss
                    total_loss.backward()
                    optimizer.step()
                    epoch_ce_loss_sum += ce_loss.item()
                    epoch_metric_loss_sum += metric_loss.item()
                    epoch_batches += 1

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
                            preds = torch.where(torch.argmax(out_occ, 1) == 0, 0, torch.argmax(out_piece, 1) + 1)
                        elif heads == 3:
                            out_occ, out_color, out_piece, _ = model(inputs)
                            p_occ, p_color, p_piece6 = torch.argmax(out_occ, 1), torch.argmax(out_color,
                                                                                              1), torch.argmax(
                                out_piece, 1)
                            preds = torch.where(p_occ == 0, 0, torch.where(p_color == 1, p_piece6 + 1, p_piece6 + 7))

                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(t_unified.cpu().numpy())

                val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
                avg_ce_loss = epoch_ce_loss_sum / max(1, epoch_batches)
                avg_triplet_loss = epoch_metric_loss_sum / max(1, epoch_batches)
                print(
                    f"   Epoch {epoch + 1}: "
                    f"Val F1={val_f1:.2f}% | "
                    f"CrossEntropy Loss={avg_ce_loss:.4f} | "
                    f"Triplet Loss ({triplet_mode})={avg_triplet_loss:.4f}"
                )
                scheduler.step(val_f1)

                if val_f1 > best_fold_f1:
                    best_fold_f1 = val_f1
                    torch.save(model.state_dict(), os.path.join(config_dir, f"best_{val_game}.pth"))

                early_stopping(val_f1)
                if early_stopping.early_stop:
                    print(f"   -> Early stopping triggered at epoch {epoch + 1} (Best F1: {best_fold_f1:.2f}%)")
                    break

            fold_f1_scores.append(best_fold_f1)

        mean_cv_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0
        print(f"\n>>> Configuration [{config_name}] Mean 4-Fold CV F1: {mean_cv_f1:.2f}%")

        # --- ENSEMBLE EVALUATION ---
        print(f">>> Running 4-Model Ensemble Evaluation on game5...")
        ensemble_test_f1 = evaluate_ensemble_on_test(config_dir, "game5", heads, freeze, device)
        print(f">>> FINAL UNSEEN TEST F1: {ensemble_test_f1:.2f}%")

        results.append({
            "Config ID": config_name,
            "Heads": heads,
            "Sampling": sampling,
            "Triplet Loss": triplet_mode,
            "Freeze Backbone": freeze,
            "Batch Size": batch_size,
            "Mean 4-Fold F1": mean_cv_f1,
            "Ensemble Test F1 (game5)": ensemble_test_f1
        })
        pd.DataFrame(results).to_csv(os.path.join(output_base, "running_results_exp.csv"), index=False)

    print(
        "\n=============================================\nALL 72 PERMUTATIONS COMPLETE!\n=============================================")
    final_df = pd.DataFrame(results).sort_values(by="Ensemble Test F1 (game5)", ascending=False)
    final_df.to_csv(os.path.join(output_base, "FINAL_RESULTS.csv"), index=False)
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()