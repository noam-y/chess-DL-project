import torch
import argparse
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import load_model_module from train_all.py
from train_all import load_model_module

def analyze(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model Protocol
    print(f"Loading model from {args.model_file}...")
    model_protocol = load_model_module(args.model_file)
    
    # Access ID_TO_PIECE from the loaded module if available
    model_module = sys.modules["model_module"]
    if hasattr(model_module, "ID_TO_PIECE"):
        id_to_piece = model_module.ID_TO_PIECE
        class_names = [id_to_piece[i] for i in range(len(id_to_piece))]
    else:
        print("Warning: ID_TO_PIECE not found in model module. Using numeric labels.")
        class_names = [str(i) for i in range(13)]

    # Create Dataset for the specific game (Validation Mode)
    print(f"Loading dataset for game: {args.game_name}...")
    val_ds = model_protocol.create_dataset(args.data_dir, mode='val', val_game_name=args.game_name)
    
    if len(val_ds) == 0:
        print(f"No samples found for game: {args.game_name}")
        return

    # Load Checkpoint
    model = model_protocol.create_model().to(device)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model_{args.game_name}.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    failures = []
    
    print(f"Analyzing {len(val_ds)} samples...")
    
    # Iterate one by one to keep track of filenames
    # (Batching is faster but makes tracking filenames harder unless we modify the dataset)
    # Given the dataset size (5888), one by one is acceptable for analysis.
    
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            sample = val_ds[i]
            if sample is None: continue
            
            # Unpack sample (image, label_tensor)
            # Note: BaseChessDataset returns (image, label_tensor) for V3
            img_tensor = sample[0]
            label_tensor = sample[1]
            
            # Add batch dimension
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            
            pred_cls = pred.item()
            true_cls = label_tensor.item()
            
            all_preds.append(pred_cls)
            all_labels.append(true_cls)
            
            if pred_cls != true_cls:
                # Get filename from dataset
                img_path, fen_char = val_ds.data[i]
                failures.append({
                    'path': img_path,
                    'true_label': fen_char,
                    'pred_label': class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls),
                    'true_id': true_cls,
                    'pred_id': pred_cls
                })

    # Generate Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12,10))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {args.game_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    output_png = f"confusion_matrix_{args.game_name}.png"
    plt.savefig(output_png)
    print(f"Confusion matrix saved to {output_png}")
    
    # Save Failures
    if failures:
        df_fail = pd.DataFrame(failures)
        output_csv = f"failures_{args.game_name}.csv"
        df_fail.to_csv(output_csv, index=False)
        print(f"Saved {len(failures)} failures to {output_csv}")
    else:
        print("No failures found! (100% accuracy)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--game_name", type=str, required=True)
    
    args = parser.parse_args()
    analyze(args)
