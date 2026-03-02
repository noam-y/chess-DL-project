import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import torchvision.models as models
import chessboard_image as cbi
from tqdm import tqdm

# Mapping to match train_v2.py
ID_TO_PIECE = {
    0: 'e', 
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}

class SmartChessNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SmartChessNet, self).__init__()
        try:
            self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except:
            self.base_model = models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.base_model(x)

def fen_from_board(board_grid):
    fen_rows = []
    for row in board_grid:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == 'e':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

def infer_tile(model, tile_tensor, device, ood_threshold=0.7):
    tile_tensor = tile_tensor.to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(tile_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        
        char = ID_TO_PIECE[pred_idx.item()]
        is_ood = conf.item() < ood_threshold
        return char, is_ood

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="unlalbled")
    parser.add_argument("--output_dir", type=str, default="results_v2")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ood_threshold", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.checkpoints_dir, args.model)
    model = SmartChessNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Matches the 96x96 resize and normalization from train_v2.py
    infer_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = glob.glob(os.path.join(args.input_dir, "*.jpg")) + glob.glob(os.path.join(args.input_dir, "*.png"))
    
    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            original_img = img.copy()
            # Standardizing input board size
            img_resized = img.resize((480, 480), resample=Image.BILINEAR)
            
            board_grid = []
            ood_mask = []
            tile_size = 60 # 480 / 8
            
            for r in range(8):
                row_pieces = []
                for c in range(8):
                    left, upper = c * tile_size, r * tile_size
                    tile_img = img_resized.crop((left, upper, left+tile_size, upper+tile_size))
                    input_tensor = infer_transform(tile_img)
                    piece, is_ood = infer_tile(model, input_tensor, device, args.ood_threshold)
                    row_pieces.append(piece)
                    if is_ood: ood_mask.append((r, c))
                board_grid.append(row_pieces)
            
            fen = fen_from_board(board_grid)
            print(f"\n{os.path.basename(img_path)}: {fen}")
            
            fen_img_path = os.path.join(args.output_dir, "temp_fen.png")
            cbi.generate_image(fen, fen_img_path, size=480, show_coordinates=False)
            
            if os.path.exists(fen_img_path):
                ood_fen_img = Image.open(fen_img_path).convert("RGB")
                draw = ImageDraw.Draw(ood_fen_img)
                cell_w, cell_h = ood_fen_img.width / 8, ood_fen_img.height / 8
                
                for r, c in ood_mask:
                    x_min, y_min = c * cell_w, r * cell_h
                    x_max, y_max = (c + 1) * cell_w, (r + 1) * cell_h
                    m_x, m_y = cell_w * 0.2, cell_h * 0.2
                    draw.line([(x_min+m_x, y_min+m_y), (x_max-m_x, y_max-m_y)], fill="red", width=5)
                    draw.line([(x_min+m_x, y_max-m_y), (x_max-m_x, y_min+m_y)], fill="red", width=5)
                
                display_img = original_img.resize((480, 480))
                combined = Image.new('RGB', (display_img.width + ood_fen_img.width, 480), (255, 255, 255))
                combined.paste(display_img, (0, 0))
                combined.paste(ood_fen_img, (display_img.width, 0))
                combined.save(os.path.join(args.output_dir, f"result_{os.path.basename(img_path)}"))
                if os.path.exists(fen_img_path): os.remove(fen_img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
