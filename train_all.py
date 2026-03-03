import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import importlib.util
import sys
import glob
import inspect
from collections import Counter
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def load_model_module(model_file_path):
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)

    found_class = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type):
            bases = [b.__name__ for b in obj.__mro__]
            if "ChessModelProtocol" in bases:
                if name in ["ChessModelProtocol", "BaseChessModel"]: continue
                if obj.__module__ != "model_module": continue
                if inspect.isabstract(obj): continue
                found_class = obj
                break
    if found_class: return found_class()
    raise ValueError("No concrete class implementing ChessModelProtocol found.")


def find_games(root_dir):
    abs_root = os.path.abspath(root_dir)
    csv_files = glob.glob(os.path.join(abs_root, '**', 'gt.csv'), recursive=True)
    found_games = set()
    for csv_path in csv_files:
        found_games.add(os.path.basename(os.path.dirname(csv_path)))
    return sorted(list(found_games))


def train_one_fold(model_protocol, args, val_game, device):
    print(f"\n{'=' * 50}")
    print(f"STARTING FOLD: Validate on {val_game}")
    print(f"{'=' * 50}")

    train_ds = model_protocol.create_dataset(args.data_dir, mode='train', val_game_name=val_game)
    val_ds = model_protocol.create_dataset(args.data_dir, mode='val', val_game_name=val_game)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    if len(train_ds) == 0: return 0.0

    try:
        collate_fn = model_protocol.get_collate_fn()
    except AttributeError:
        from torch.utils.data import default_collate
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            return default_collate(batch) if len(batch) > 0 else None

    # The 50/50 Uniform Sampler
    if hasattr(train_ds, 'all_labels') and len(train_ds.all_labels) > 0:
        labels = train_ds.all_labels
        class_counts = Counter(labels)

        sample_weights = []
        for label in labels:
            if label == 'e':
                # Empty squares share 50% of the selection probability
                weight = 0.5 / max(1, class_counts['e'])
            else:
                # The 12 piece classes uniformly share the remaining 50%
                weight = (0.5 / 12.0) / max(1, class_counts[label])

            sample_weights.append(weight)

        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)

        sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn,
                                  num_workers=4)
        print(f"50/50 WeightedRandomSampler initialized. Found {len(class_counts)} classes.")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=4)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = model_protocol.create_model().to(device)
    optimizer = model_protocol.get_optimizer(model, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_fold_f1 = 0.0
    best_report = ""
    best_cm = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fold {val_game} Epoch {epoch + 1}", leave=False):
            if batch is None: continue
            optimizer.zero_grad()
            loss, _ = model_protocol.compute_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue

                # Protocol agnostic validation: extract metrics directly from compute_loss
                loss, metrics = model_protocol.compute_loss(model, batch, device)
                val_loss_sum += loss.item()

                all_preds.extend(metrics['preds'])
                all_targets.extend(metrics['targets'])

        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else 0

        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        val_acc = (np.array(all_preds) == np.array(all_targets)).mean() * 100

        scheduler.step(val_loss)

        if val_f1 > best_fold_f1:
            best_fold_f1 = val_f1
            best_report = classification_report(all_targets, all_preds, zero_division=0)
            best_cm = confusion_matrix(all_targets, all_preds)
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{val_game}.pth"))

        print(f"Epoch {epoch + 1}: Train Loss={epoch_loss:.3f}, Val Acc={val_acc:.1f}%, Val Macro F1={val_f1:.2f}%")

    print(f"\n--- Best Validation Results for Fold {val_game} ---")
    print(f"Best Macro F1: {best_fold_f1:.2f}%")
    print("Classification Report:\n", best_report)
    print("Confusion Matrix:\n", best_cm)

    return best_fold_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        args.output_dir = f"./checkpoints_{model_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    model_protocol = load_model_module(args.model_file)
    all_games = find_games(args.data_dir)

    if not all_games: return

    results = {}
    for game in all_games:
        acc = train_one_fold(model_protocol, args, game, device)
        results[game] = acc

    f1_scores = list(results.values())
    print(f"\nFinal K-Fold Mean Macro F1-Score: {np.mean(f1_scores):.2f}% ± {np.std(f1_scores):.2f}%")


if __name__ == "__main__":
    main()