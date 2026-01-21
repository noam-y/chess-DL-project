import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

# Adding current dir to path to avoid import errors
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}
ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}
ID_TO_PIECE[0] = '1'

# --- 1. Model Definition (Must match train_v2.py exactly) ---
class SmartChessNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNet, self).__init__()
        try:
            self.base_model = models.resnet18(weights=None) 
        except:
            self.base_model = models.resnet18(pretrained=False)
            
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

# --- 2. Helper Functions ---
def prediction_to_fen(pred_tensor):
    """ Converts the 8x8 prediction tensor to a FEN string """
    board = pred_tensor.cpu().numpy()
    rows_fen = []
    for r in range(8):
        row_str = ""
        empty_count = 0
        for c in range(8):
            val = board[r, c]
            char = ID_TO_PIECE.get(val, '1')
            if char == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += char
        if empty_count > 0:
            row_str += str(empty_count)
        rows_fen.append(row_str)
    return "/".join(rows_fen)

def compare_fens(true_fen, pred_fen):
    """ Compares Ground Truth FEN with Predicted FEN """
    def expand(fen):
        rows = fen.split(' ')[0].split('/')
        res = []
        for row in rows:
            for char in row:
                if char.isdigit():
                    res.extend(['.'] * int(char))
                else:
                    res.append(char)
        return res

    list_true = expand(true_fen)
    list_pred = expand(pred_fen)
    
    if len(list_pred) != 64: return 0, False

    correct_count = sum([1 for t, p in zip(list_true, list_pred) if t == p])
    is_perfect = (correct_count == 64)
    return correct_count, is_perfect

# --- 3. Smart Dataset for Evaluation ---
class SmartEvalDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = os.path.abspath(root_dir)
        
        # *** MUST MATCH TRAIN_V2 ***
        # If you changed target_size in train_v2.py, change it here too!
        self.target_size = 96 
        
        self.resize_transform = transforms.Resize((self.target_size, self.target_size))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def crop_square(self, image, row, col):
        """ Contextual Cropping Logic """
        width, height = image.size
        square_w = width / 8
        square_h = height / 8
        
        center_x = col * square_w + square_w / 2
        center_y = row * square_h + square_h / 2
        
        crop_size = square_w * 1.5 
        
        x1 = max(0, center_x - crop_size / 2)
        y1 = max(0, center_y - crop_size / 2)
        x2 = min(width, center_x + crop_size / 2)
        y2 = min(height, center_y + crop_size / 2)
        
        return image.crop((x1, y1, x2, y2))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['from_frame']
        img_path = os.path.join(self.root_dir, img_name)
        
        if not os.path.exists(img_path):
            # Fallback if filename is just the name but full path needed
            pass

        image = Image.open(img_path).convert('RGB')
        
        patches = []
        for r in range(8):
            for c in range(8):
                patch = self.crop_square(image, r, c)
                patch = self.resize_transform(patch)
                patch = self.transform(patch)
                patches.append(patch)
        
        return torch.stack(patches), row['fen'], row['filename']

# --- 4. Main Evaluation Loop ---
def main(args):
    # Setup paths
    csv_path = os.path.join(args.test_dir, args.csv_name)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        return

    print(f"Model: {args.model_path}")
    print(f"Data:  {args.test_dir}")
    print(f"Device: {DEVICE}")

    # Load Dataset
    dataset = SmartEvalDataset(csv_path, args.test_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Load Model
    model = SmartChessNet(num_classes=13).to(DEVICE)
    
    try:
        state_dict = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    total_squares = 0
    correct_squares = 0
    total_boards = 0
    perfect_boards = 0
    results = [] 

    print("Running Inference...")
    with torch.no_grad():
        for images, true_fens, filenames in tqdm(loader):
            Batch_Size = images.shape[0]
            inputs = images.view(-1, 3, 96, 96).to(DEVICE) # Ensure size matches target_size
            
            outputs = model(inputs)
            preds_flat = torch.argmax(outputs, dim=1) 
            preds_grid = preds_flat.view(Batch_Size, 8, 8)
            
            for i in range(Batch_Size):
                pred_fen_str = prediction_to_fen(preds_grid[i])
                true_fen_str = true_fens[i]
                
                correct, is_perfect = compare_fens(true_fen_str, pred_fen_str)
                
                total_squares += 64
                correct_squares += correct
                total_boards += 1
                if is_perfect:
                    perfect_boards += 1
                
                results.append({
                    'filename': filenames[i],
                    'true_fen': true_fen_str,
                    'pred_fen': pred_fen_str,
                    'accuracy': correct / 64.0,
                    'is_perfect': is_perfect
                })

    piece_acc = 100 * correct_squares / total_squares if total_squares > 0 else 0
    board_acc = 100 * perfect_boards / total_boards if total_boards > 0 else 0
    
    print("-" * 30)
    print(f"Piece Accuracy: {piece_acc:.2f}%")
    print(f"Board Accuracy: {board_acc:.2f}%")
    print("-" * 30)

    output_csv = "evaluation_results.csv"
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default="aug/new_augmented_data")
    parser.add_argument("--csv_name", type=str, default="augmented_ground_truth.csv")

    args = parser.parse_args()
    main(args)