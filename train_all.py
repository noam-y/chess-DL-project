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
                if name in ["ChessModelProtocol", "BaseChessModel"]:
                    continue
                if obj.__module__ != "model_module":
                    continue
                if inspect.isabstract(obj):
                    continue
                found_class = obj
                break
    
    if found_class:
        return found_class()
    else:
        raise ValueError("No concrete class implementing ChessModelProtocol found.")

def find_games(root_dir):
    abs_root = os.path.abspath(root_dir)
    csv_files = glob.glob(os.path.join(abs_root, '**', 'gt.csv'), recursive=True)
    found_games = set()
    for csv_path in csv_files:
        game_dir = os.path.dirname(csv_path)
        game_name = os.path.basename(game_dir)
        found_games.add(game_name)
    return sorted(list(found_games))

def train_one_fold(model_protocol, args, val_game, device):
    print(f"\n{'='*40}")
    print(f"STARTING FOLD: Validate on {val_game}")
    print(f"{'='*40}")

    train_ds = model_protocol.create_dataset(args.data_dir, mode='train', val_game_name=val_game)
    val_ds = model_protocol.create_dataset(args.data_dir, mode='val', val_game_name=val_game)
    
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        return 0.0

    try:
        collate_fn = model_protocol.get_collate_fn()
    except AttributeError:
        from torch.utils.data import default_collate
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if len(batch) == 0: return None
            return default_collate(batch)

    # ========================================================
    # Sampler Configuration
    # ========================================================
    if args.sampler == 'weighted' and hasattr(train_ds, 'all_labels') and len(train_ds.all_labels) > 0:
        labels = train_ds.all_labels
        class_counts = Counter(labels)
        
        print("\nClass distribution in Training Set:")
        for k, v in sorted(class_counts.items()):
            print(f"  {k}: {v} samples")
            
        sample_weights = [1.0 / class_counts[label] for label in labels]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights_tensor, 
            num_samples=len(sample_weights_tensor), 
            replacement=True
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate_fn, num_workers=4
        )
        print(f"WeightedRandomSampler initialized. Found {len(class_counts)} unique classes.")
    else:
        if args.sampler == 'weighted':
            print("Warning: 'all_labels' not found in dataset. Falling back to standard shuffling.")
        else:
            print("Using standard shuffling (no sampler).")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = model_protocol.create_model().to(device)
    optimizer = model_protocol.get_optimizer(model, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_fold_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Fold {val_game} Epoch {epoch+1}", leave=False):
            if batch is None: continue
            
            optimizer.zero_grad()
            loss, metrics = model_protocol.compute_loss(model, batch, device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_correct += metrics['correct']
            train_total += metrics['total']

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                _, metrics = model_protocol.compute_loss(model, batch, device)
                val_correct += metrics['correct']
                val_total += metrics['total']
        
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        scheduler.step(epoch_loss)
        
        if val_acc > best_fold_acc:
            best_fold_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{val_game}.pth"))

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.3f}, Train Acc={train_acc:.1f}%, Val Acc ({val_game})={val_acc:.1f}%")

    print(f"Finished Fold {val_game}. Best Val Acc: {best_fold_acc:.2f}%")
    return best_fold_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sampler", type=str, default="weighted", choices=["weighted", "none"],
                        help="Sampler to use for training data ('weighted' or 'none').")
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        args.output_dir = f"./checkpoints_{model_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)

    model_protocol = load_model_module(args.model_file)
    all_games = find_games(args.data_dir)
    
    if not all_games:
        return

    results = {}
    for game in all_games:
        acc = train_one_fold(model_protocol, args, game, device)
        results[game] = acc
    
    accuracies = list(results.values())
    print(f"\nFinal K-Fold Mean Acc: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")

if __name__ == "__main__":
    main()