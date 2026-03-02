nirnir
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util
import sys
import glob
import pandas as pd
from torchvision import transforms
from PIL import Image
import inspect

def load_model_module(model_file_path):
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    
    # Find the class that implements ChessModelProtocol
    # We need to find a class that inherits from BaseChessModel (or ChessModelProtocol)
    # AND is NOT BaseChessModel itself (which is abstract).
    
    found_class = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type):
            # Check if it inherits from ChessModelProtocol (directly or indirectly)
            bases = [b.__name__ for b in obj.__mro__]
            if "ChessModelProtocol" in bases:
                # Skip the abstract base classes themselves
                if name in ["ChessModelProtocol", "BaseChessModel"]:
                    continue
                
                # CRITICAL FIX: Only pick classes defined IN THIS MODULE
                # This prevents picking up BaseChessModel which is imported
                if obj.__module__ != "model_module":
                    continue
                
                # Ensure it's not abstract
                if inspect.isabstract(obj):
                    continue

                found_class = obj
                break
    
    if found_class:
        return found_class()
    else:
        # Fallback: print what was found to help debug
        print(f"Debug: Classes found in {model_file_path}:")
        for name, obj in module.__dict__.items():
            if isinstance(obj, type):
                print(f" - {name} (Module: {obj.__module__})")
        raise ValueError("No concrete class implementing ChessModelProtocol found in the provided file.")

def find_games(root_dir):
    abs_root = os.path.abspath(root_dir)
    csv_files = glob.glob(os.path.join(abs_root, '**', '*.csv'), recursive=True)
    found_games = set()
    for csv_path in csv_files:
        path_parts = csv_path.split(os.sep)
        current_game = next((part for part in path_parts if 'game' in part.lower()), None)
        if current_game:
            found_games.add(current_game)
    return sorted(list(found_games))

def train_one_fold(model_protocol, args, val_game, device):
    print(f"\n{'='*40}")
    print(f"STARTING FOLD: Validate on {val_game}")
    print(f"{'='*40}")

    train_ds = model_protocol.create_dataset(args.data_dir, mode='train', val_game_name=val_game)
    val_ds = model_protocol.create_dataset(args.data_dir, mode='val', val_game_name=val_game)
    
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Use default_collate if get_collate_fn is not available or returns None
    try:
        collate_fn = model_protocol.get_collate_fn()
    except AttributeError:
        from torch.utils.data import default_collate
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if len(batch) == 0: return None
            return default_collate(batch)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = model_protocol.create_model().to(device)
    optimizer = model_protocol.get_optimizer(model, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_fold_acc = 0.0

    for epoch in range(args.epochs):
        # --- TRAIN ---
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

        # --- VALIDATION ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                _, metrics = model_protocol.compute_loss(model, batch, device)
                val_correct += metrics['correct']
                val_total += metrics['total']
        
        epoch_loss = running_loss / len(train_loader)
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
    parser.add_argument("--model_file", type=str, required=True, help="Path to the python file implementing the model protocol")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints. Defaults to ./checkpoints_{model_name}")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # Set default output_dir based on model_file name if not provided
    if args.output_dir is None:
        model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        args.output_dir = f"./checkpoints_{model_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    model_protocol = load_model_module(args.model_file)
    
    # Use the generic find_games function instead of the protocol method
    all_games = find_games(args.data_dir)
    
    if not all_games:
        print("Warning: Auto-detection of games failed or no games found.")
        return

    print(f"Found games for Cross-Validation: {all_games}")
    
    results = {}
    
    for game in all_games:
        acc = train_one_fold(model_protocol, args, game, device)
        results[game] = acc
    
    print("\n" + "="*40)
    print("FINAL K-FOLD RESULTS")
    print("="*40)
    accuracies = []
    for game, acc in results.items():
        print(f"Hold-out {game}: {acc:.2f}%")
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("-" * 40)
    print(f"Average Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
