import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIECE_TYPES_INV = {0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k'}

class SmartChessNetV3(nn.Module):
    def __init__(self):
        super(SmartChessNetV3, self).__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.head_occ = nn.Linear(num_ftrs, 2)
        self.head_color = nn.Linear(num_ftrs, 2)
        self.head_piece = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        return self.head_occ(features), self.head_color(features), self.head_piece(features)

def predictions_to_fen(pred_occ, pred_color, pred_piece):
    pred_occ = pred_occ.cpu().numpy()
    pred_color = pred_color.cpu().numpy()
    pred_piece = pred_piece.cpu().numpy()
    
    rows_fen = []
    for r in range(8):
        row_str, empty_count = "", 0
        for c in range(8):
            idx = r * 8 + c
            if pred_occ[idx] == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                
                char = PIECE_TYPES_INV[pred_piece[idx]]
                if pred_color[idx] == 1:
                    char = char.upper()
                row_str += char
                
        if empty_count > 0:
            row_str += str(empty_count)
        rows_fen.append(row_str)
    return "/".join(rows_fen)

def compare_fens(true_fen, pred_fen):
    def expand(f):
        res = []
        parts = f.split(' ')[0].split('/')
        for row in parts:
            for char in row:
                if char.isdigit():
                    res.extend(['.'] * int(char))
                else:
                    res.append(char)
        return res
    t_list, p_list = expand(true_fen), expand(pred_fen)
    if len(p_list) != 64: return 0, False
    correct = sum([1 for t, p in zip(t_list, p_list) if t == p])
    return correct, (correct == 64)

class SmartEvalDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.df.columns = self.df.columns.str.strip()
        self.root_dir = os.path.abspath(root_dir)
        self.images_dir = self.root_dir if not os.path.exists(os.path.join(self.root_dir, 'tagged_images')) else os.path.join(self.root_dir, 'tagged_images')
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def crop_square(self, image, row, col):
        w, h = image.size
        sw, sh = w/8, h/8
        cx, cy, cs = col*sw + sw/2, row*sh + sh/2, sw * 1.5
        return image.crop((max(0, cx-cs/2), max(0, cy-cs/2), min(w, cx+cs/2), min(h, cy+cs/2)))

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename'] if 'filename' in row else f"frame_{int(row['from_frame']):06d}.jpg"
        img_path = os.path.join(self.images_dir, img_name)
        if not os.path.exists(img_path): return torch.zeros(64, 3, 96, 96), row['fen'], "missing"
        image = Image.open(img_path).convert('RGB')
        patches = [self.transform(self.crop_square(image, r, c)) for r in range(8) for c in range(8)]
        return torch.stack(patches), row['fen'], img_name

def main(args):
    csv_path = os.path.join(args.test_dir, args.csv_name)
    dataset = SmartEvalDataset(csv_path, args.test_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = SmartChessNetV3().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    
    t_sq, c_sq, t_bd, p_bd, results = 0, 0, 0, 0, []
    
    with torch.no_grad():
        for imgs, true_fens, fnames in tqdm(loader):
            imgs = imgs.squeeze(0)
            if imgs.shape[0] != 64: continue
            
            out_occ, out_color, out_piece = model(imgs.to(DEVICE))
            
            p_occ = torch.argmax(out_occ, dim=1)
            p_color = torch.argmax(out_color, dim=1)
            p_piece = torch.argmax(out_piece, dim=1)
            
            pf = predictions_to_fen(p_occ, p_color, p_piece)
            tf = true_fens[0]
            
            c, p = compare_fens(tf, pf)
            t_sq += 64; c_sq += c; t_bd += 1
            if p: p_bd += 1
            
            results.append({'filename': fnames[0], 'true_fen': tf, 'pred_fen': pf, 'accuracy': c/64.0, 'is_perfect': p})
            
    if t_sq > 0:
        print(f"\nPiece Accuracy: {100*c_sq/t_sq:.2f}%\nBoard Accuracy: {100*p_bd/t_bd if t_bd > 0 else 0:.2f}%")
        pd.DataFrame(results).to_csv("evaluation_results_v3.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--csv_name", required=True)
    main(parser.parse_args())
