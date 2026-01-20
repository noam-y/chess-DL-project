import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# --- 1. ייבוא המודל שלך ---
# חשוב: החליפי את 'ChessModel' בשם המחלקה האמיתי של המודל שלך מתוך train.py
# ואת 'train' בשם הקובץ בו מוגדר המודל (אם הוא שונה)
try:
    from train import ChessModel 
except ImportError:
    # הגדרת דמי למקרה שהייבוא נכשל, רק כדי שהקוד ירוץ להדגמה
    print("Warning: Could not import ChessModel from train.py. Using dummy placeholder.")
    class ChessModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # מניחים פלט של (Batch, 13 classes, 8 rows, 8 cols)
            self.conv = torch.nn.Conv2d(3, 13, kernel_size=1) 
        def forward(self, x):
            # מחזיר טנזור בגודל (B, 13, 8, 8) - לכל משבצת יש 13 סבירויות
            return torch.randn(x.shape[0], 13, 8, 8)

# --- 2. הגדרות ומיפוי הפוך ---
TEST_DIR = 'new_augmented_data'
CSV_PATH = os.path.join(TEST_DIR, 'augmented_ground_truth.csv')
MODEL_PATH = 'best_model.pth' # הנתיב לקובץ המשקולות ששמרת
IMG_SIZE = 480
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# בניית המילון ההפוך על סמך train.py
PIECE_TO_ID = {
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}
# מילון הפוך: מספר -> אות. 0 ממופה לנקודה (ריק)
ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}
ID_TO_PIECE[0] = '1' # ב-FEN משתמשים במספרים לריק, נטפל בזה בפונקציה

# --- 3. פונקציות עזר ---
def prediction_to_fen(pred_tensor):
    """
    מקבל טנזור של אינדקסים בגודל (8, 8)
    וממיר אותו למחרוזת FEN חוקית.
    """
    # המרה ל-numpy
    board = pred_tensor.cpu().numpy()
    rows_fen = []
    
    for r in range(8):
        row_str = ""
        empty_count = 0
        for c in range(8):
            val = board[r, c]
            char = ID_TO_PIECE.get(val, '1') # ברירת מחדל '1' אם לא נמצא
            
            if char == '1': # משבצת ריקה
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
    """
    משווה בין שני FENs ברמת המשבצת.
    מחזיר: (מספר משבצות נכונות, האם הלוח מושלם)
    """
    # פונקציה פנימית לפריסת FEN לרשימה שטוחה של 64 תווים
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
    
    # בדיקה ששניהם באורך 64 (קריטי)
    if len(list_pred) != 64: 
        # במקרה של חיזוי לא חוקי באורכו
        return 0, False

    correct_count = sum([1 for t, p in zip(list_true, list_pred) if t == p])
    is_perfect = (correct_count == 64)
    
    return correct_count, is_perfect

# --- 4. Dataset ---
class EvalDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        # אותן טרנספורמציות כמו ב-train.py!
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, row['fen'], row['filename']

# --- 5. Main Loop ---
def main():
    # א. טעינת הנתונים
    dataset = EvalDataset(CSV_PATH, TEST_DIR)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # ב. אתחול המודל
    print(f"Loading model from {MODEL_PATH}...")
    model = ChessModel().to(DEVICE)
    
    # ניסיון לטעון משקולות (חובה שיהיה קובץ pth)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Warning: Model weights file not found! Running with random weights.")

    model.eval() # מצב הערכה (מכבה Dropout וכו')

    total_squares = 0
    correct_squares = 0
    total_boards = 0
    perfect_boards = 0
    
    results = []

    print("Starting evaluation...")
    with torch.no_grad():
        for images, true_fens, filenames in tqdm(loader):
            images = images.to(DEVICE)
            
            # 1. Forward Pass
            outputs = model(images) 
            # Output Shape Assumption: (Batch, 13, 8, 8)
            # אנחנו רוצים לקחת את המחלקה עם הסבירות הגבוהה ביותר לכל פיקסל/משבצת
            preds = torch.argmax(outputs, dim=1) # Shape: (Batch, 8, 8)
            
            # 2. Decoding & Comparison
            for i in range(len(filenames)):
                pred_grid = preds[i] # טנזור בגודל 8x8
                pred_fen_str = prediction_to_fen(pred_grid)
                true_fen_str = true_fens[i]
                
                correct, is_perfect = compare_fens(true_fen_str, pred_fen_str)
                
                total_squares += 64
                correct_squares += correct
                total_boards += 1
                if is_perfect:
                    perfect_boards += 1
                
                # שמירה ל-CSV תוצאות (אופציונלי)
                results.append({
                    'filename': filenames[i],
                    'true_fen': true_fen_str,
                    'pred_fen': pred_fen_str,
                    'accuracy': correct / 64.0
                })

    # ג. סיכום
    piece_acc = 100 * correct_squares / total_squares
    board_acc = 100 * perfect_boards / total_boards
    
    print("\n" + "="*40)
    print(f"EVALUATION RESULTS")
    print("="*40)
    print(f"Total Images Evaluated: {total_boards}")
    print(f"Piece-wise Accuracy:    {piece_acc:.2f}%  (כמה כלים בודדים זוהו נכון)")
    print(f"Board-wise Accuracy:    {board_acc:.2f}%  (כמה לוחות זוהו מושלם 100%)")
    print("="*40)

    # שמירת קובץ שגיאות לניתוח
    res_df = pd.DataFrame(results)
    res_df.to_csv("evaluation_results_detail.csv", index=False)
    print("Detailed results saved to 'evaluation_results_detail.csv'")

if __name__ == "__main__":
    main()