# --- UPDATED EXPERIMENT 2 WITH LOGGING AND DISCRIMINATIVE LR ---
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from collections import Counter
from sklearn.metrics import f1_score
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


# [Existing mappings and MultiSimilarityLoss class from previous turn remain same]

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


def get_optimizer(model, freeze_mode):
    if freeze_mode == 'improved':
        # Split parameters into groups for discriminative LR
        return optim.Adam([
            {'params': model.backbone.parameters(), 'lr': 1e-5},  # Low LR for backbone
            {'params': model.fc.parameters(), 'lr': 1e-4}  # High LR for head
        ])
    return optim.Adam(model.parameters(), lr=1e-4)


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... [Dataset and Sampler setup remains same] ...

    results = []
    for f_mode, t_mode, b_size in itertools.product(freeze_opts, triplet_opts, batch_opts):
        # ... [Inner loop start] ...
        model = Experiment2Model(f_mode).to(device)
        optimizer = get_optimizer(model, f_mode)

        running_ce_loss = []
        running_metric_loss = []

        for epoch in range(15):
            model.train()
            for batch in train_loader:
                # ... [Pre-processing] ...
                logits, feats = model(imgs)
                ce_loss = nn.CrossEntropyLoss()(logits, targets)

                # Metric Loss calculation
                if t_mode == 'batch_hard':
                    # [Batch Hard Logic]
                    metric_loss = calculate_batch_hard(feats, targets)
                else:
                    metric_loss = ms_loss_fn(feats, targets)

                # LOGGING: Save individual components
                running_ce_loss.append(ce_loss.item())
                running_metric_loss.append(metric_loss.item())

                optimizer.zero_grad()
                (ce_loss + metric_loss).backward()
                optimizer.step()

        results.append({
            "Config": config_id,
            "Avg_CE_Loss": np.mean(running_ce_loss),
            "Avg_Metric_Loss": np.mean(running_metric_loss),
            "Mean_F1": np.mean(fold_scores)
        })
        pd.DataFrame(results).to_csv("experiment2_results.csv", index=False)