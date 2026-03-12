#!/usr/bin/env python3
"""Evaluate saved fold models on a held-out test game."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from train import (
    ConfigurableChessResNet,
    GridDataset,
    chars_to_tensors,
    custom_collate,
    infer_unified_probs,
)

IDX_TO_LABEL = {
    0: "empty",
    1: "white_pawn",
    2: "white_knight",
    3: "white_bishop",
    4: "white_rook",
    5: "white_queen",
    6: "white_king",
    7: "black_pawn",
    8: "black_knight",
    9: "black_bishop",
    10: "black_rook",
    11: "black_queen",
    12: "black_king",
}


def resolve_checkpoint_dir(config_dir: Path) -> Path:
    """Accept either the fold-checkpoint directory or its parent directory."""
    if not config_dir.exists():
        raise FileNotFoundError(f"Directory not found: {config_dir}")

    if any(config_dir.glob("*.pth")):
        return config_dir

    candidate_dirs = [d for d in config_dir.iterdir() if d.is_dir() and any(d.glob("*.pth"))]
    if not candidate_dirs:
        raise FileNotFoundError(f"No .pth files found under: {config_dir}")
    if len(candidate_dirs) > 1:
        dirs_text = ", ".join(str(d) for d in candidate_dirs)
        raise ValueError(
            "Multiple candidate checkpoint directories found. "
            f"Please pass --config-dir explicitly. Candidates: {dirs_text}"
        )
    return candidate_dirs[0]


def evaluate_ensemble_on_test(
    config_dir: Path,
    test_game_name: str,
    device: torch.device,
    test_aug: bool = True,
) -> dict[str, object]:
    data_dir = "assets/dataset"
    test_ds = GridDataset(data_dir, mode="test", test_game=test_game_name)
    if len(test_ds) == 0:
        return {
            "macro_f1": 0.0,
            "confusion_matrix": np.zeros((13, 13), dtype=int),
            "per_piece": [],
            "num_samples": 0,
        }

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate,
    )

    models_list = []
    for ckpt in sorted(config_dir.glob("*.pth")):
        model = ConfigurableChessResNet().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        models_list.append(model)

    if not models_list:
        return {
            "macro_f1": 0.0,
            "confusion_matrix": np.zeros((13, 13), dtype=int),
            "per_piece": [],
            "num_samples": 0,
        }

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            boards, chars = batch
            inputs = boards.to(device)
            t_unified, _, _ = chars_to_tensors(chars, device)

            ensemble_probs = torch.zeros((inputs.size(0), 13), device=device)
            for model in models_list:
                ensemble_probs += infer_unified_probs(model, inputs, test_aug=test_aug)

            ensemble_probs /= len(models_list)
            all_preds.extend(torch.argmax(ensemble_probs, dim=1).cpu().numpy())
            all_targets.extend(t_unified.cpu().numpy())

    labels = list(range(13))
    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100
    precision, recall, f1_each, support = precision_recall_fscore_support(
        all_targets,
        all_preds,
        labels=labels,
        zero_division=0,
    )

    per_piece = []
    for idx in labels:
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        per_piece.append(
            {
                "label": IDX_TO_LABEL[idx],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1_each[idx]),
            }
        )

    return {
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_piece": per_piece,
        "num_samples": len(all_targets),
    }


def print_confusion_matrix(cm: np.ndarray) -> None:
    labels = [IDX_TO_LABEL[i] for i in range(13)]
    header = "true\\pred".ljust(14) + " ".join(label[:3].rjust(5) for label in labels)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(header)
    for i, row in enumerate(cm):
        row_name = labels[i][:12].ljust(14)
        row_values = " ".join(str(int(v)).rjust(5) for v in row)
        print(f"{row_name}{row_values}")


def print_per_piece_report(per_piece: list[dict[str, object]]) -> None:
    print("\nPer-piece metrics:")
    print("label".ljust(14), "TP".rjust(5), "FP".rjust(5), "FN".rjust(5), "prec".rjust(8), "rec".rjust(8), "f1".rjust(8))
    for m in per_piece:
        print(
            str(m["label"]).ljust(14),
            str(m["tp"]).rjust(5),
            str(m["fp"]).rjust(5),
            str(m["fn"]).rjust(5),
            f'{float(m["precision"]):.3f}'.rjust(8),
            f'{float(m["recall"]):.3f}'.rjust(8),
            f'{float(m["f1"]):.3f}'.rjust(8),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained fold models on a test game.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("train_results"),
        help="Fold checkpoints directory, or parent directory that contains it.",
    )
    parser.add_argument(
        "--test-game",
        type=str,
        default="game5",
        help="Test game folder name (default: game5).",
    )
    parser.add_argument(
        "--test-aug",
        action="store_true",
        help="Enable test-time augmentation for evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_dir = resolve_checkpoint_dir(args.config_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = evaluate_ensemble_on_test(
        config_dir=resolved_dir,
        test_game_name=args.test_game,
        device=device,
        test_aug=args.test_aug,
    )

    print(f"Checkpoint dir: {resolved_dir}")
    print(f"Test game: {args.test_game}")
    print(f"Test-time augmentation: {args.test_aug}")
    print(f"Samples: {results['num_samples']}")
    print(f"Ensemble Test F1 ({args.test_game}): {results['macro_f1']:.2f}%")
    print_confusion_matrix(results["confusion_matrix"])
    print_per_piece_report(results["per_piece"])


if __name__ == "__main__":
    main()
