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
import torchvision.transforms.functional as TF

# --- GLOBAL MAPPINGS ---
CHAR_TO_UNIFIED = {'e': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                   'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
CHAR_TO_PIECE12 = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                   'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
CHAR_TO_PIECE6 = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}


class AddGaussianNoise:
    def __init__(self, sigma_min=0.0, sigma_max=0.02, p=0.1):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            sigma = torch.empty(1).uniform_(self.sigma_min, self.sigma_max).item()
            noise = torch.randn_like(tensor) * sigma
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


class RandomGaussianBlur:
    def __init__(self, p=0.1):
        self.p = p
        self.blur3 = transforms.GaussianBlur(kernel_size=3)
        self.blur5 = transforms.GaussianBlur(kernel_size=5)

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            if torch.rand(1).item() < 0.5:
                return self.blur3(img)
            return self.blur5(img)
        return img


class RandomBlackenEdges:
    def __init__(self, max_ratio=0.25):
        self.max_ratio = max_ratio

    def __call__(self, img):
        arr = np.array(img)
        h, w = arr.shape[:2]

        top = int(torch.empty(1).uniform_(0.0, self.max_ratio).item() * h)
        bottom = int(torch.empty(1).uniform_(0.0, self.max_ratio).item() * h)
        left = int(torch.empty(1).uniform_(0.0, self.max_ratio).item() * w)
        right = int(torch.empty(1).uniform_(0.0, self.max_ratio).item() * w)

        if top > 0:
            arr[:top, :, :] = 0
        if bottom > 0:
            arr[h - bottom:, :, :] = 0
        if left > 0:
            arr[:, :left, :] = 0
        if right > 0:
            arr[:, w - right:, :] = 0

        return Image.fromarray(arr)


# --- CUSTOM ROBUST DATASET ---
class GridDataset(Dataset):
    def __init__(self, data_dir, mode='train', val_game=None, test_game='game5', data_aug=False):
        self.data_dir = data_dir
        self.samples = []
        self.all_labels = []

        if mode == 'train' and data_aug:
            self.transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.05, p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                RandomBlackenEdges(max_ratio=0.25),
                RandomGaussianBlur(p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.05),
                transforms.ToTensor(),
                AddGaussianNoise(sigma_min=0.0, sigma_max=0.02, p=0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
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
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        num_ftrs = resnet.fc.in_features
        if self.num_heads == 1: self.head_main = nn.Linear(num_ftrs, 13)
        elif self.num_heads == 2:
            self.head_occ = nn.Linear(num_ftrs, 2)
            self.head_piece = nn.Linear(num_ftrs, 12)
        elif self.num_heads == 3:
            self.head_occ = nn.Linear(num_ftrs, 2)
            self.head_white_piece = nn.Linear(num_ftrs, 6)
            self.head_black_piece = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        if self.num_heads == 1: return self.head_main(features), features
        elif self.num_heads == 2: return self.head_occ(features), self.head_piece(features), features
        elif self.num_heads == 3:
            return self.head_occ(features), self.head_white_piece(features), self.head_black_piece(features), features

# --- EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0):
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
            if label == 'e': sample_weights.append(0.5 / max(1, class_counts['e']))
            else: sample_weights.append((0.5 / 12.0) / max(1, class_counts[label]))
    tensor_weights = torch.tensor(sample_weights, dtype=torch.float)
    return WeightedRandomSampler(tensor_weights, len(tensor_weights), replacement=True)

def chars_to_tensors(chars, device):
    t_unified = torch.tensor([CHAR_TO_UNIFIED[c] for c in chars], dtype=torch.long, device=device)
    t_occ = torch.tensor([0 if c == 'e' else 1 for c in chars], dtype=torch.long, device=device)
    t_color = torch.tensor([0 if c == 'e' else (1 if c.isupper() else 0) for c in chars], dtype=torch.long, device=device)
    t_piece12 = torch.tensor([0 if c == 'e' else CHAR_TO_PIECE12[c] for c in chars], dtype=torch.long, device=device)
    t_piece6 = torch.tensor([0 if c == 'e' else CHAR_TO_PIECE6[c.lower()] for c in chars], dtype=torch.long, device=device)
    return t_unified, t_occ, t_color, t_piece12, t_piece6


def denormalize_batch(inputs):
    mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, 3, 1, 1)
    return torch.clamp(inputs * std + mean, 0.0, 1.0)


def normalize_batch(inputs):
    mean = torch.tensor([0.485, 0.456, 0.406], device=inputs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=inputs.device).view(1, 3, 1, 1)
    return (inputs - mean) / std


def blacken_edges_tensor(inputs, edge_ratio=0.25):
    out = inputs.clone()
    _, _, h, w = out.shape
    top = int(h * edge_ratio)
    bottom = int(h * edge_ratio)
    left = int(w * edge_ratio)
    right = int(w * edge_ratio)
    if top > 0:
        out[:, :, :top, :] = 0.0
    if bottom > 0:
        out[:, :, h - bottom:, :] = 0.0
    if left > 0:
        out[:, :, :, :left] = 0.0
    if right > 0:
        out[:, :, :, w - right:] = 0.0
    return out


def build_test_time_aug_batches(inputs):
    denorm = denormalize_batch(inputs)

    aug_color_rot = []
    for img in denorm:
        brightness = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
        contrast = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
        saturation = 1.0 + float(torch.empty(1).uniform_(-0.05, 0.05))
        hue = float(torch.empty(1).uniform_(-0.01, 0.01))
        angle = float(torch.empty(1).uniform_(-3.0, 3.0))

        img_aug = TF.adjust_brightness(img, brightness)
        img_aug = TF.adjust_contrast(img_aug, contrast)
        img_aug = TF.adjust_saturation(img_aug, saturation)
        img_aug = TF.adjust_hue(img_aug, hue)
        img_aug = TF.rotate(
            img_aug,
            angle=angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0.0
        )
        aug_color_rot.append(torch.clamp(img_aug, 0.0, 1.0))

    aug_color_rot = normalize_batch(torch.stack(aug_color_rot, dim=0))
    aug_hflip = normalize_batch(torch.flip(denorm, dims=[3]))
    return [inputs, aug_color_rot, aug_hflip]


def unified_probs_from_outputs(model, inputs, heads):
    if heads == 1:
        out_main, _ = model(inputs)
        return F.softmax(out_main, dim=1)
    if heads == 2:
        out_occ, out_piece, _ = model(inputs)
        prob_occ = F.softmax(out_occ, dim=1)
        prob_piece = F.softmax(out_piece, dim=1)
        unified_prob = torch.zeros((inputs.size(0), 13), device=inputs.device)
        unified_prob[:, 0] = prob_occ[:, 0]
        for i in range(12):
            unified_prob[:, i + 1] = prob_occ[:, 1] * prob_piece[:, i]
        return unified_prob

    out_occ, out_white_piece, out_black_piece, _ = model(inputs)
    prob_occ = F.softmax(out_occ, dim=1)
    prob_white_piece = F.softmax(out_white_piece, dim=1)
    prob_black_piece = F.softmax(out_black_piece, dim=1)
    unified_prob = torch.zeros((inputs.size(0), 13), device=inputs.device)
    unified_prob[:, 0] = prob_occ[:, 0]
    for i in range(6):
        unified_prob[:, i + 1] = prob_occ[:, 1] * 0.5 * prob_white_piece[:, i]
        unified_prob[:, i + 7] = prob_occ[:, 1] * 0.5 * prob_black_piece[:, i]
    return unified_prob


def infer_unified_probs(model, inputs, heads, test_aug=False):
    if not test_aug:
        return unified_probs_from_outputs(model, inputs, heads)

    tta_inputs = build_test_time_aug_batches(inputs)
    tta_probs = [unified_probs_from_outputs(model, aug_inputs, heads) for aug_inputs in tta_inputs]
    return torch.stack(tta_probs, dim=0).mean(dim=0)


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


def focal_loss(logits, targets, gamma=2.0, alpha=None, label_smoothing=0.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce_loss)
    focal_term = (1 - pt) ** gamma
    if alpha is not None:
        alpha_t = alpha.to(logits.device)[targets]
        focal_term = alpha_t * focal_term
    return (focal_term * ce_loss).mean()


def classification_loss(loss_name, logits, targets, smoothing=False, smoothing_eps=0.05):
    label_smoothing = smoothing_eps if smoothing else 0.0
    if loss_name == "focal":
        return focal_loss(logits, targets, label_smoothing=label_smoothing)
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def apply_freezing_schedule(model, freezing, epoch_num):
    if not freezing:
        for p in model.parameters():
            p.requires_grad = True
        return

    # Heads are always trainable.
    for name, p in model.named_parameters():
        if not name.startswith("backbone."):
            p.requires_grad = True

    # Epochs 1-5: freeze entire backbone.
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Epochs 6-10: unfreeze last two backbone blocks (layer3/layer4).
    if 6 <= epoch_num <= 10:
        for block_idx in [6, 7]:
            for p in model.backbone[block_idx].parameters():
                p.requires_grad = True

    # Epoch 11+: unfreeze everything.
    if epoch_num >= 11:
        for p in model.backbone.parameters():
            p.requires_grad = True


def build_optimizer(model, freezing, head_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4):
    if not freezing:
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=head_lr,
            weight_decay=weight_decay
        )

    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    backbone_params = list(model.backbone.parameters())
    return optim.Adam(
        [
            {"params": head_params, "lr": head_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ],
        weight_decay=weight_decay
    )


# --- ENSEMBLE EVALUATOR ---
def evaluate_ensemble_on_test(config_dir, test_game_name, heads, device, test_aug=False):
    data_dir = "assets/new_dataset"
    test_ds = GridDataset(data_dir, mode='test', test_game=test_game_name)
    if len(test_ds) == 0: return 0.0
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

    models_list = []
    for file in os.listdir(config_dir):
        if file.endswith(".pth"):
            model = ConfigurableChessResNet(heads).to(device)
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
                ensemble_probs += infer_unified_probs(model, inputs, heads, test_aug=test_aug)

            ensemble_probs /= len(models_list)
            all_preds.extend(torch.argmax(ensemble_probs, dim=1).cpu().numpy())
            all_targets.extend(t_unified.cpu().numpy())

    return f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100


# --- MAIN GRID SEARCH ---
def main():
    data_dir = "assets/new_dataset"
    output_base = "experiment114_results_occ_white_black_nofreeze"
    os.makedirs(output_base, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting grid search on {device}...")

    heads_opts = [2, 3]
    sample_opts = ['50_50']
    triplet_opts = ['new', 'old']
    batch_size_opts = [32]
    loss_opts = ['ce', 'focal']
    smoothing_opts = [True, False]
    freezing_opts = [True, False]
    data_aug_opts = [True, False]
    test_aug_opts = [True, False]


    all_combinations = list(
        itertools.product(
            heads_opts,
            sample_opts,
            triplet_opts,
            batch_size_opts,
            loss_opts,
            smoothing_opts,
            freezing_opts,
            data_aug_opts,
            test_aug_opts
        )
    )
    todo = all_combinations

    all_games = ['game2', 'game4', 'game6', 'game7']
    results = []

    for combination_idx, (heads, sampling, triplet_mode, batch_size, loss_name, smoothing, freezing, data_aug, test_aug) in enumerate(todo, 1):
        config_name = (
            f"H{heads}_S-{sampling}_T-{triplet_mode}_B-{batch_size}_L-{loss_name}_SM-{smoothing}_F-{freezing}_DA-{data_aug}_TA-{test_aug}"
        )
        print(f"\n{'=' * 60}\nModel {combination_idx}/{len(todo)}: {config_name}\n{'=' * 60}")

        config_dir = os.path.join(output_base, config_name)
        os.makedirs(config_dir, exist_ok=True)
        fold_f1_scores = []
        fold_epochs = {}
        fold_scores = {}

        for val_game in all_games:
            print(f"\n--- Fold: Validating on {val_game} ---")
            train_ds = GridDataset(data_dir, mode='train', val_game=val_game, data_aug=data_aug)
            val_ds = GridDataset(data_dir, mode='val', val_game=val_game)
            if len(train_ds) == 0:
                print(f"Skipping {val_game} (No training data found)")
                continue

            sampler = get_sampler(sampling, train_ds.all_labels)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None),
                                      num_workers=4, collate_fn=custom_collate)
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

            model = ConfigurableChessResNet(heads).to(device)
            apply_freezing_schedule(model, freezing, epoch_num=1)
            optimizer = build_optimizer(model, freezing, head_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
            early_stopping = EarlyStopping(patience=40)

            best_fold_f1 = 0.0
            final_epoch = 0

            # MAIN EVENT: Up to 100 epochs
            for epoch in range(100):
                current_epoch = epoch + 1
                apply_freezing_schedule(model, freezing, epoch_num=current_epoch)
                final_epoch = epoch + 1
                model.train()
                for batch in tqdm(train_loader, desc=f"Epoch {current_epoch} Train", leave=False):
                    if batch is None: continue
                    boards, chars = batch
                    inputs = boards.to(device)
                    t_unified, t_occ, t_color, t_piece12, t_piece6 = chars_to_tensors(chars, device)

                    optimizer.zero_grad()
                    total_loss = torch.tensor(0.0, device=device)
                    mask = (t_occ == 1)

                    if heads == 1:
                        out_main, features = model(inputs)
                        total_loss += classification_loss(loss_name, out_main, t_unified, smoothing=smoothing)
                        if triplet_mode == "old": total_loss += calculate_triplet_loss(features, t_unified, mask, device)
                        else: total_loss += calculate_multi_similarity_loss(features, t_piece12, mask, device)
                    elif heads == 2:
                        out_occ, out_piece, features = model(inputs)
                        total_loss += classification_loss(loss_name, out_occ, t_occ, smoothing=smoothing)
                        if mask.sum() > 0:
                            total_loss += classification_loss(
                                loss_name, out_piece[mask], t_piece12[mask], smoothing=smoothing
                            )
                        if triplet_mode == "old": total_loss += calculate_triplet_loss(features, t_piece12, mask, device)
                        else: total_loss += calculate_multi_similarity_loss(features, t_piece12, mask, device)
                    elif heads == 3:
                        out_occ, out_white_piece, out_black_piece, features = model(inputs)
                        mask_white = mask & (t_color == 1)
                        mask_black = mask & (t_color == 0)
                        total_loss += classification_loss(loss_name, out_occ, t_occ, smoothing=smoothing)
                        if mask_white.sum() > 0:
                            total_loss += classification_loss(
                                loss_name, out_white_piece[mask_white], t_piece6[mask_white], smoothing=smoothing
                            )
                        if mask_black.sum() > 0:
                            total_loss += classification_loss(
                                loss_name, out_black_piece[mask_black], t_piece6[mask_black], smoothing=smoothing
                            )
                        if triplet_mode == "old":
                            total_loss += calculate_triplet_loss(features, t_piece6, mask_white, device)
                            total_loss += calculate_triplet_loss(features, t_piece6, mask_black, device)
                        else:
                            total_loss += calculate_multi_similarity_loss(features, t_piece6, mask_white, device)
                            total_loss += calculate_multi_similarity_loss(features, t_piece6, mask_black, device)


                    total_loss.backward()
                    optimizer.step()

                model.eval()
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        if batch is None: continue
                        boards, chars = batch
                        inputs = boards.to(device)
                        t_unified, _, _, _, _ = chars_to_tensors(chars, device)
                        # Keep validation fast/consistent for scheduler + early stopping.
                        # TTA is only used in the final saved-model evaluation.
                        unified_probs = infer_unified_probs(model, inputs, heads, test_aug=False)
                        preds = torch.argmax(unified_probs, dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(t_unified.cpu().numpy())

                val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
                print(
                    f"   Epoch {current_epoch}: "
                    f"Val F1={val_f1:.2f}% | "
                )
                scheduler.step(val_f1)

                if val_f1 > best_fold_f1:
                    best_fold_f1 = val_f1
                    torch.save(model.state_dict(), os.path.join(config_dir, f"best_{val_game}.pth"))

                early_stopping(val_f1)
                if early_stopping.early_stop:
                    print(f"   -> Early stopping triggered at epoch {current_epoch} (Best F1: {best_fold_f1:.2f}%)")
                    break

            fold_epochs[val_game] = final_epoch
            fold_scores[val_game] = best_fold_f1
            fold_f1_scores.append(best_fold_f1)

        mean_cv_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0
        print(f"\n>>> Configuration [{config_name}] Mean 4-Fold CV F1: {mean_cv_f1:.2f}%")

        # --- ENSEMBLE EVALUATION ---
        print(f">>> Running 4-Model Ensemble Evaluation on game5...")
        ensemble_test_f1 = evaluate_ensemble_on_test(config_dir, "game5", heads, device, test_aug=test_aug)
        print(f">>> FINAL UNSEEN TEST F1: {ensemble_test_f1:.2f}%")

        results.append({
            "Config ID": config_name,
            "Heads": heads,
            "Sampling": sampling,
            "Triplet Loss": triplet_mode,
            "Batch Size": batch_size,
            "Loss": loss_name,
            "Label Smoothing": smoothing,
            "Freezing": freezing,
            "Data Augmentation": data_aug,
            "Test-Time Augmentation": test_aug,
            "Mean 4-Fold F1": mean_cv_f1,
            "Ensemble Test F1 (game5)": ensemble_test_f1,
            "Epochs game2": fold_epochs.get('game2', 0),
            "Best F1 game2": fold_scores.get('game2', 0),
            "Epochs game4": fold_epochs.get('game4', 0),
            "Best F1 game4": fold_scores.get('game4', 0),
            "Epochs game6": fold_epochs.get('game6', 0),
            "Best F1 game6": fold_scores.get('game6', 0),
            "Epochs game7": fold_epochs.get('game7', 0),
            "Best F1 game7": fold_scores.get('game7', 0)
        })
        pd.DataFrame(results).to_csv(os.path.join(output_base, "running_results_exp_114.csv"), index=False)

    print(
        f"\n=============================================\nALL {len(todo)} CONFIGURATIONS COMPLETE!\n=============================================")
    final_df = pd.DataFrame(results).sort_values(by="Ensemble Test F1 (game5)", ascending=False)
    final_df.to_csv(os.path.join(output_base, "FINAL_RESULTS_114.csv"), index=False)
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()