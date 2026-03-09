import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE_MAP = {1: 'p', 2: 'n', 3: 'b', 4: 'r', 5: 'q', 6: 'k'}

class MultiHeadPieceClassifier(nn.Module):
    def __init__(self):
        super(MultiHeadPieceClassifier, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_dim = 128 * 7 * 7
        self.head_occ = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        self.head_color = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 3))
        self.head_type = nn.Sequential(nn.Flatten(), nn.Linear(self.fc_dim, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):
        features = self.backbone(x)
        return self.head_occ(features), self.head_color(features), self.head_type(features)

def compare_fens(true_fen, pred_fen):
    def fen_to_list(f):
        res = []
        board = f.split(' ')[0]
        for row in board.split('/'):
            for char in row:
                if char.isdigit(): res.extend(['1'] * int(char))
                else: res.append(char)
        return res
    
    t_list, p_list = fen_to_list(true_fen), fen_to_list(pred_fen)
    correct = sum(1 for t, p in zip(t_list, p_list) if t == p)
    return correct, (correct == 64)

def heads_to_fen(p_occ, p_color, p_type):
    occ = torch.argmax(p_occ, 1).cpu().numpy()
    color = torch.argmax(p_color, 1).cpu().numpy()
    ptype = torch.argmax(p_type, 1).cpu().numpy()
    
    res_board = []
    for i in range(64):
        if occ[i] == 0: # Empty
            res_board.append('1')
        else:
            p_char = TYPE_MAP.get(ptype[i], 'p')
            res_board.append(p_char.upper() if color[i] == 1 else p_char.lower())
    # Convert list back to FEN format
    rows = []
    for r in range(8):
        row_chars = res_board[r*8:(r+1)*8]
        row_str, empty = "", 0
        for c in row_chars:
            if c == '1': empty += 1
            else:
                if empty > 0: row_str += str(empty); empty = 0
                row_str += c
        if empty > 0: row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows)

class ChessEvalDataset(Dataset):
    def __init__(self, root_dir, csv_name):
        self.root_dir = root_dir
        self.df = pd.read_csv(os.path.join(root_dir, csv_name))
        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_id = int(row['from_frame'])
        img_name = f"frame_{frame_id:06d}.jpg"
        img_path = os.path.join(self.root_dir, 'tagged_images', img_name)
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        patches = image.unfold(1, 60, 60).unfold(2, 60, 60)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, 60, 60)
        
        return patches, row['fen'], img_name

def main(args):
    model = MultiHeadPieceClassifier().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded Multi-Head model from {args.model_path}")

    dataset = ChessEvalDataset(args.test_dir, args.csv_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    total_sq, correct_sq, perfect_boards = 0, 0, 0

    with torch.no_grad():
        for patches, true_fen, img_name in tqdm(loader):
            patches = patches.view(-1, 3, 60, 60).to(DEVICE)
            p_occ, p_color, p_type = model(patches)
            
            pred_fen = heads_to_fen(p_occ, p_color, p_type)
            correct, is_perfect = compare_fens(true_fen[0], pred_fen)
            
            correct_sq += correct
            total_sq += 64
            if is_perfect: perfect_boards += 1
            
            results.append({'file': img_name[0], 'acc': correct/64, 'perfect': is_perfect})

    print(f"\nPiece Accuracy: {100*correct_sq/total_sq:.2f}%")
    print(f"Board Accuracy: {100*perfect_boards/len(dataset):.2f}%")
    pd.DataFrame(results).to_csv("multihead_eval_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--csv_name", type=str, required=True)
    main(parser.parse_args())