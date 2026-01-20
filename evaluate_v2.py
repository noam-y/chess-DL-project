import os
import sys
import argparse
import torch
import torch.nn as nn # הוספנו
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models # הוספנו
from PIL import Image
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# --- הגדרות ---
IMG_SIZE = 480
PATCH_SIZE = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}
ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}
ID_TO_PIECE[0] = '1'

# --- המודל החדש (חייב להיות זהה ל-train.py) ---
class SmartChessNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNet, self).__init__()
        
        # אנחנו לא צריכים weights=DEFAULT כאן כי אנחנו טוענים משקולות מהקובץ שלך
        try:
            self.base_model = models.resnet18(weights=None) 
        except:
            self.base_model = models.resnet18(pretrained=False)
            
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

def prediction_to_fen(pred_tensor):
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

class EvalDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # --- עדכון חשוב: Normalization ---
        # ResNet חייב את הנרמול הזה כדי לעבוד כמו שצריך
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, row['fen'], row['filename']

def main(args):
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

    dataset = EvalDataset(csv_path, args.test_dir)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # שימוש במודל החדש
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
            images = images.to(DEVICE)
            Batch_Size = images.shape[0]

            # Cut images into patches
            patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE)
            
            # Inference
            outputs = model(patches)
            
            # קבלת האינדקס (הכלי) בעל ההסתברות הגבוהה ביותר
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
    parser.add_argument("--test_dir", type=str, default="aug/new_augmented_data") # עדכנתי את דיפולט
    parser.add_argument("--csv_name", type=str, default="augmented_ground_truth.csv")

    args = parser.parse_args()
    main(args)