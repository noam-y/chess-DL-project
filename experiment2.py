import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, default_collate
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


# --- DATASET ---
class GridDataset(Dataset):
    def __init__(self, data_dir, mode='train', val_game=None, test_game='game5'):
        self.data_dir = data_dir
        self.samples, self.all_labels = [], []
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        games = [d for d in os.listdir(data_dir) if d.startswith('game') and os.path.isdir(os.path.join(data_dir, d))]
        for game in games:
            if game == test_game and mode != 'test': continue
            if mode == 'train' and game == val_game: continue
            if mode == 'val' and game != val_game: continue
            csv_path = os.path.join(data_dir, game, 'gt.csv')
            if not os.path.exists(csv_path): continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.samples.append((os.path.join(data_dir, game, 'tagged_images', row['file_name']), row['fen']))
                self.all_labels.append(row['fen'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, char = self.samples[idx]
        try:
            return self.transform(Image.open(img_path).convert('RGB')), char
        except:
            return None


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch) if batch else None


# --- MODELS & LOSSES ---
class Experiment2Model(nn.Module):
    def __init__(self, freeze_mode):
        super().__init__()
        self.freeze_mode = freeze_mode
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 13)
        if self.freeze_mode == 'improved':
            for param in self.backbone.parameters(): param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_mode == 'improved':
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()

    def forward(self, x):
        feats = torch.flatten(self.backbone(x), 1)
        return self.fc(feats), feats


class MultiSimilarityLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.thresh, self.margin, self.scale_pos, self.scale_neg = 0.5, 0.1, 2.0, 40.0
        self.device = device

    def forward(self, feats, labels):
        batch_size = feats.shape[0]
        sim_mat = torch.matmul(feats, feats.t())
        loss = []
        for i in range(batch_size):
            pos_idx = (labels == labels[i]);
            pos_idx[i] = 0
            neg_idx = (labels != labels[i])
            if not pos_idx.any() or not neg_idx.any(): continue
            pos_sim, neg_sim = sim_mat[i][pos_idx], sim_mat[i][neg_idx]
            neg_algo = neg_sim[neg_sim + self.margin > torch.min(pos_sim)]
            pos_algo = pos_sim[pos_sim - self.margin < torch.max(neg_sim)]
            if not neg_algo.any() or not pos_algo.any(): continue
            l_p = (1.0 / self.scale_pos) * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_algo - self.thresh))))
            l_n = (1.0 / self.scale_neg) * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_algo - self.thresh))))
            loss.append(l_p + l_n)
        return torch.stack(loss).mean() if loss else torch.tensor(0.0, device=self.device)


def calculate_batch_hard(feats, labels, device):
    mask = (labels != 0)
    if mask.sum() < 2 or len(labels[mask].unique()) < 2: return torch.tensor(0.0, device=device)
    dist = torch.cdist(feats[mask], feats[mask])
    lbls = labels[mask].unsqueeze(0) == labels[mask].unsqueeze(1)
    hp = (dist * lbls.float()).max(1)[0]
    hn = (dist + (dist.max() + 1) * (~lbls).float()).min(1)[0]
    return F.relu(1.0 + hp - hn).mean()


def get_optimizer(model, freeze_mode):
    if freeze_mode == 'improved':
        return optim.Adam([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.fc.parameters(), 'lr': 1e-4}
        ])
    return optim.Adam(model.parameters(), lr=1e-4)


# --- MAIN EXPERIMENT ---
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "assets/new_dataset"

    freeze_opts = ['none', 'improved']
    triplet_opts = ['batch_hard', 'multi_similarity']
    batch_opts = [16, 32, 64]
    all_games = ['game2', 'game4', 'game6', 'game7']

    results = []
    ms_loss_fn = MultiSimilarityLoss(device)

    for f_mode, t_mode, b_size in itertools.product(freeze_opts, triplet_opts, batch_opts):
        config_id = f"F-{f_mode}_T-{t_mode}_B-{b_size}"
        print(f"\n>>> Running Config: {config_id}")

        fold_f1s = []
        running_ce_loss = []
        running_metric_loss = []

        for val_game in all_games:
            train_ds = GridDataset(data_dir, mode='train', val_game=val_game)
            val_ds = GridDataset(data_dir, mode='val', val_game=val_game)

            counts = Counter(train_ds.all_labels)
            weights = [(0.5 / counts['e'] if c == 'e' else 0.5 / 12.0 / counts[c]) for c in train_ds.all_labels]
            train_loader = DataLoader(train_ds, batch_size=b_size, sampler=WeightedRandomSampler(weights, len(weights)),
                                      collate_fn=custom_collate)
            val_loader = DataLoader(val_ds, batch_size=64, collate_fn=custom_collate)

            model = Experiment2Model(f_mode).to(device)
            optimizer = get_optimizer(model, f_mode)

            best_f1 = 0
            for epoch in range(15):
                model.train()
                for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                    if not batch: continue
                    imgs, chars = batch
                    imgs, targets = imgs.to(device), torch.tensor([CHAR_TO_UNIFIED[c] for c in chars], device=device)

                    optimizer.zero_grad()
                    logits, feats = model(imgs)
                    ce_loss = nn.CrossEntropyLoss()(logits, targets)

                    m_loss = calculate_batch_hard(feats, targets, device) if t_mode == 'batch_hard' else ms_loss_fn(
                        feats, targets)

                    running_ce_loss.append(ce_loss.item())
                    running_metric_loss.append(m_loss.item())

                    (ce_loss + m_loss).backward()
                    optimizer.step()

                model.eval()
                preds, targs = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        if not batch: continue
                        out, _ = model(batch[0].to(device))
                        preds.extend(torch.argmax(out, 1).cpu().numpy())
                        targs.extend([CHAR_TO_UNIFIED[c] for c in batch[1]])
                f1 = f1_score(targs, preds, average='macro', zero_division=0) * 100
                if f1 > best_f1: best_f1 = f1

            fold_f1s.append(best_f1)

        results.append({
            "Config": config_id,
            "Mean_F1": np.mean(fold_f1s),
            "Avg_CE_Loss": np.mean(running_ce_loss),
            "Avg_Metric_Loss": np.mean(running_metric_loss)
        })
        pd.DataFrame(results).to_csv("experiment2_results.csv", index=False)


if __name__ == "__main__":
    run_experiment()