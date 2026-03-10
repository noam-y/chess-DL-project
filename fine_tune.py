import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from experiment114 import (
    GridDataset,
    custom_collate,
    ConfigurableChessResNet,
    apply_freezing_schedule,
    build_optimizer,
    EarlyStopping,
    get_sampler,
    chars_to_tensors,
    classification_loss,
    calculate_triplet_loss,
    calculate_multi_similarity_loss,
    infer_unified_probs,
    evaluate_ensemble_on_test,
)


BASE_CONFIG = {
    "heads": 2,
    "sampling": "50_50",
    "triplet_mode": "new",
    "batch_size": 64,
    "loss_name": "focal",
    "smoothing": True,
    "freezing": False,
    "data_aug": True,
    "test_aug": True,
    "metric_weight": 0.5,
}

OFAT_SWEEP_VALUES = {
    "heads": [1, 3],
    "triplet_mode": ["none", "old"],
    "batch_size": [32, 16],
    "loss_name": ["ce"],
    "smoothing": [False],
    "freezing": [True],
    "data_aug": [False],
    "test_aug": [False],
    "metric_weight": [0.25, 0.75],
    "sampling": ["uniform", "none"],
}

CONFIG_ORDER = [
    "heads",
    "triplet_mode",
    "batch_size",
    "loss_name",
    "smoothing",
    "data_aug",
    "test_aug",
    "metric_weight",
    "sampling",
    "freezing",
]


def build_ofat_todo(base_config, sweep_values):
    """Build one-factor-at-a-time configs: baseline + one changed parameter."""
    todo = [(base_config.copy(), "baseline", base_config.copy())]
    seen = {config_name_from_dict(base_config)}

    for param in CONFIG_ORDER:
        values = sweep_values.get(param, [])
        for value in values:
            if value == base_config[param]:
                continue
            cfg = base_config.copy()
            cfg[param] = value
            cfg_name = config_name_from_dict(cfg)
            if cfg_name in seen:
                continue
            seen.add(cfg_name)
            todo.append((cfg, param, value))
    return todo


def config_name_from_dict(cfg):
    return (
        f"H{cfg['heads']}_S-{cfg['sampling']}_T-{cfg['triplet_mode']}_B-{cfg['batch_size']}"
        f"_L-{cfg['loss_name']}_SM-{cfg['smoothing']}_F-{cfg['freezing']}_DA-{cfg['data_aug']}"
        f"_TA-{cfg['test_aug']}_MW-{cfg['metric_weight']}"
    )


def run_one_config(cfg, data_dir, output_base, all_games, device, changed_param="baseline", changed_value=None):
    config_name = config_name_from_dict(cfg)
    print(f"\n{'=' * 60}\nRunning: {config_name}\n{'=' * 60}")

    config_dir = os.path.join(output_base, config_name)
    os.makedirs(config_dir, exist_ok=True)
    fold_f1_scores = []
    fold_epochs = {}
    fold_scores = {}

    for val_game in all_games:
        print(f"\n--- Fold: Validating on {val_game} ---")
        train_ds = GridDataset(data_dir, mode="train", val_game=val_game, data_aug=cfg["data_aug"])
        val_ds = GridDataset(data_dir, mode="val", val_game=val_game)
        if len(train_ds) == 0:
            print(f"Skipping {val_game} (No training data found)")
            continue

        sampler = get_sampler(cfg["sampling"], train_ds.all_labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            collate_fn=custom_collate,
        )
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

        model = ConfigurableChessResNet(cfg["heads"]).to(device)
        apply_freezing_schedule(model, cfg["freezing"], epoch_num=1)
        optimizer = build_optimizer(model, cfg["freezing"], head_lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=2)
        early_stopping = EarlyStopping()

        best_fold_f1 = 0.0
        final_epoch = 0

        for epoch in range(100):
            current_epoch = epoch + 1
            final_epoch = current_epoch
            apply_freezing_schedule(model, cfg["freezing"], epoch_num=current_epoch)
            model.train()

            for batch in tqdm(train_loader, desc=f"Epoch {current_epoch} Train", leave=False):
                if batch is None:
                    continue
                boards, chars = batch
                inputs = boards.to(device)
                t_unified, t_occ, t_color, t_piece12, t_piece6 = chars_to_tensors(chars, device)
                mask = t_occ == 1

                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=device)

                if cfg["heads"] == 1:
                    out_main, features = model(inputs)
                    total_loss += classification_loss(cfg["loss_name"], out_main, t_unified, smoothing=cfg["smoothing"])
                    if cfg["triplet_mode"] == "old":
                        total_loss += cfg["metric_weight"] * calculate_triplet_loss(features, t_unified, mask, device)
                    elif cfg["triplet_mode"] == "new":
                        total_loss += cfg["metric_weight"] * calculate_multi_similarity_loss(features, t_piece12, mask, device)
                elif cfg["heads"] == 2:
                    out_occ, out_piece, features = model(inputs)
                    total_loss += classification_loss(cfg["loss_name"], out_occ, t_occ, smoothing=cfg["smoothing"])
                    if mask.sum() > 0:
                        total_loss += classification_loss(cfg["loss_name"], out_piece[mask], t_piece12[mask], smoothing=cfg["smoothing"])
                    if cfg["triplet_mode"] == "old":
                        total_loss += cfg["metric_weight"] * calculate_triplet_loss(features, t_piece12, mask, device)
                    elif cfg["triplet_mode"] == "new":
                        total_loss += cfg["metric_weight"] * calculate_multi_similarity_loss(features, t_piece12, mask, device)
                elif cfg["heads"] == 3:
                    out_occ, out_white_piece, out_black_piece, features = model(inputs)
                    mask_white = mask & (t_color == 1)
                    mask_black = mask & (t_color == 0)
                    total_loss += classification_loss(cfg["loss_name"], out_occ, t_occ, smoothing=cfg["smoothing"])
                    if mask_white.sum() > 0:
                        total_loss += classification_loss(cfg["loss_name"], out_white_piece[mask_white], t_piece6[mask_white], smoothing=cfg["smoothing"])
                    if mask_black.sum() > 0:
                        total_loss += classification_loss(cfg["loss_name"], out_black_piece[mask_black], t_piece6[mask_black], smoothing=cfg["smoothing"])
                    if cfg["triplet_mode"] == "old":
                        total_loss += cfg["metric_weight"] * calculate_triplet_loss(features, t_piece6, mask_white, device)
                        total_loss += cfg["metric_weight"] * calculate_triplet_loss(features, t_piece6, mask_black, device)
                    elif cfg["triplet_mode"] == "new":
                        total_loss += cfg["metric_weight"] * calculate_multi_similarity_loss(features, t_piece6, mask_white, device)
                        total_loss += cfg["metric_weight"] * calculate_multi_similarity_loss(features, t_piece6, mask_black, device)

                total_loss.backward()
                optimizer.step()

            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    boards, chars = batch
                    inputs = boards.to(device)
                    t_unified, _, _, _, _ = chars_to_tensors(chars, device)
                    unified_probs = infer_unified_probs(model, inputs, cfg["heads"], test_aug=False)
                    preds = torch.argmax(unified_probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(t_unified.cpu().numpy())

            val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100
            print(f"   Epoch {current_epoch}: Val F1={val_f1:.2f}%")
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

    mean_cv_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0.0
    print(f"\n>>> [{config_name}] Mean 4-Fold CV F1: {mean_cv_f1:.2f}%")
    ensemble_test_f1 = evaluate_ensemble_on_test(config_dir, "game5", cfg["heads"], device, test_aug=cfg["test_aug"])
    print(f">>> FINAL UNSEEN TEST F1: {ensemble_test_f1:.2f}%")

    row = {
        "Config ID": config_name,
        "Changed Param": changed_param,
        "Changed Value": changed_value,
        "Mean 4-Fold F1": mean_cv_f1,
        "Ensemble Test F1 (game5)": ensemble_test_f1,
    }
    for k in CONFIG_ORDER:
        row[k] = cfg[k]
    for g in all_games:
        row[f"Epochs {g}"] = fold_epochs.get(g, 0)
        row[f"Best F1 {g}"] = fold_scores.get(g, 0)
    return row


def main():
    data_dir = "assets/new_dataset"
    output_base = "fine_tune_results"
    os.makedirs(output_base, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tune run on {device}")
    print(f"Base config: {BASE_CONFIG}")
    print(f"OFAT sweep values: {OFAT_SWEEP_VALUES}")

    todo = build_ofat_todo(BASE_CONFIG, OFAT_SWEEP_VALUES)
    all_games = ["game2", "game4", "game6", "game7"]

    results = []
    for cfg, changed_param, changed_value in todo:
        results.append(
            run_one_config(
                cfg,
                data_dir,
                output_base,
                all_games,
                device,
                changed_param=changed_param,
                changed_value=changed_value,
            )
        )
    df = pd.DataFrame(results).sort_values(by="Ensemble Test F1 (game5)", ascending=False)
    df.to_csv(os.path.join(output_base, "FINAL_FINE_TUNE_RESULTS.csv"), index=False)
    print("\nDone. Saved:", os.path.join(output_base, "FINAL_FINE_TUNE_RESULTS.csv"))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
