import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib.util
import sys
import inspect
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def load_model_module(model_file_path):
    """Safely loads the concrete ChessModelProtocol class."""
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)

    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Look for a class that implements the protocol but is not abstract
        if name != "ChessModelProtocol" and "ChessModelProtocol" in [b.__name__ for b in obj.__mro__]:
            if not inspect.isabstract(obj) and obj.__module__ == "model_module":
                return obj()

    raise ValueError("No concrete class implementing ChessModelProtocol found.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fully trained model on a dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the test dataset directory")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the model protocol file (e.g., model_vMichael5(2head).py)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pth weights file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Protocol & Model
    print(f"Loading model architecture from {args.model_file}...")
    model_protocol = load_model_module(args.model_file)
    model = model_protocol.create_model().to(device)

    # 2. Load Weights
    print(f"Loading weights from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()  # CRITICAL: Put model in evaluation mode

    # 3. Load Dataset
    print(f"Loading test dataset from {args.data_dir}...")
    # Using mode='test' bypasses the train/val split logic and loads everything in the folder
    test_ds = model_protocol.create_dataset(args.data_dir, mode='test')
    print(f"Total test samples found: {len(test_ds)}")

    if len(test_ds) == 0:
        print("No data found. Exiting.")
        return

    try:
        collate_fn = model_protocol.get_collate_fn()
    except AttributeError:
        from torch.utils.data import default_collate
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            return default_collate(batch) if len(batch) > 0 else None

    # DataLoader (No sampler needed, we want to evaluate the raw, true distribution)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 4. Evaluation Loop
    all_preds = []
    all_targets = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if batch is None: continue

            # Use the protocol to do the heavy lifting. We ignore the loss value itself.
            _, metrics = model_protocol.compute_loss(model, batch, device)

            # The dual-head model maps its predictions to the unified 0-12 space in these arrays
            all_preds.extend(metrics['preds'])
            all_targets.extend(metrics['targets'])

    # 5. Calculate and Print Final Metrics
    val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
    val_acc = (np.array(all_preds) == np.array(all_targets)).mean() * 100

    report = classification_report(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Global Accuracy: {val_acc:.2f}%")
    print(f"Macro F1-Score:  {val_f1:.2f}%\n")

    print("--- Classification Report ---")
    print(report)

    print("--- Confusion Matrix ---")
    print(cm)
    print("=" * 50)


if __name__ == "__main__":
    main()