import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import chessboard_image as cbi
from tqdm import tqdm

# Mapping to match the single-head ID logic
ID_TO_PIECE = {
    0: 'e', 1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
}

class PieceClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(PieceClassifier, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.fc_input_dim = 128 * 7 * 7
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)

def fen_from_board(board_grid):
    fen_rows = []
    for row in board_grid:
        empty_count = 0
        fen_row = ""
        for cell in row:
            if cell == 'e': empty_count += 1
            else:
                if empty_count > 0: fen_row += str(empty_count); empty_count = 0
                fen_row += cell
        if empty_count > 0: fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="assets/unlalbled")
    parser.add_argument("--output_dir", type=str, default="results_kfold")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--conf_threshold", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = PieceClassifier(num_classes=13).to(device)
    # Loading the weight file from the --model argument
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    preprocess = transforms.Compose([transforms.Resize((480, 480)), transforms.ToTensor()])
    image_files = glob.glob(os.path.join(args.input_dir, "*.jpg")) + glob.glob(os.path.join(args.input_dir, "*.png"))
    print(f"Loaded model: {args.model}. Found {len(image_files)} images.")
    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = preprocess(img).to(device)
            patches = tensor.unfold(1, 60, 60).unfold(2, 60, 60).permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, 60, 60)

            with torch.no_grad():
                outputs = model(patches)
                probs = F.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, 1)

            board_grid, ood_mask = [], []
            preds = preds.view(8, 8)
            confs = confs.view(8, 8)

            for r in range(8):
                row = []
                for c in range(8):
                    char = ID_TO_PIECE[preds[r, c].item()]
                    row.append(char)
                    if confs[r, c].item() < args.conf_threshold: ood_mask.append((r, c))
                board_grid.append(row)

            fen = fen_from_board(board_grid)
            temp = os.path.join(args.output_dir, "temp.png")
            cbi.generate_image(fen, temp, size=480, show_coordinates=False)
            
            if os.path.exists(temp):
                fen_img = Image.open(temp).convert("RGB")
                draw = ImageDraw.Draw(fen_img)
                for r_i, c_i in ood_mask:
                    x, y = c_i * 60, r_i * 60
                    draw.line([(x+10, y+10), (x+50, y+50)], fill="red", width=3)
                    draw.line([(x+10, y+50), (x+50, y+10)], fill="red", width=3)

                res = Image.new('RGB', (960, 480))
                res.paste(img.resize((480, 480)), (0, 0))
                res.paste(fen_img, (480, 0))
                res.save(os.path.join(args.output_dir, f"out_{os.path.basename(img_path)}"))
                if os.path.exists(temp): os.remove(temp)
        except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    main()
