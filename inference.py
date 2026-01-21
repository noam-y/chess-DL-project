import os
import glob
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
import chessboard_image as cbi

# Import models from their respective files to avoid duplication
try:
    from train import PieceClassifier
except ImportError:
    print("Warning: Could not import PieceClassifier from train.py")

try:
    from nir_train import ResNetMultiHead, ResNetWithEmbeddings, ID_TO_PIECE
except ImportError:
    print("Warning: Could not import ResNetMultiHead/ResNetWithEmbeddings from nir_train.py")
    # Fallback ID mapping if import fails
    PIECE_TO_ID = {
        'e': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

# --- Utils ---

def get_latest_epoch_model(checkpoints_dir):
    # Pattern: model_epoch_{epoch}.pth OR resnet18_triplet_epoch_{epoch}.pth
    files = glob.glob(os.path.join(checkpoints_dir, "*_epoch_*.pth"))
    if not files:
        return None, None
    
    # Extract epochs
    epoch_files = []
    for f in files:
        match = re.search(r"_epoch_(\d+)\.pth", f)
        if match:
            epoch_files.append((int(match.group(1)), f))
    
    if not epoch_files:
        return None, None
        
    # Sort by epoch descending
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    return epoch_files[0] # (epoch, filepath)

def fen_from_board(board_grid):
    # board_grid is 8x8 array of piece chars
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

def infer_tile(model, tile_tensor, device, model_type, centroids=None, ood_threshold=1.0):
    tile_tensor = tile_tensor.to(device).unsqueeze(0) # (1, 3, H, W)
    
    with torch.no_grad():
        if model_type == "ResNetMultiHead":
            logits_piece, logits_color, embedding = model(tile_tensor)
            probs_piece = F.softmax(logits_piece, dim=1)
            probs_color = F.softmax(logits_color, dim=1)

            conf_piece, pred_piece_idx = torch.max(probs_piece, 1)
            pred_label = pred_piece_idx.item()

            _, pred_color_idx = torch.max(probs_color, 1)
            pred_color_label = pred_color_idx.item()  # 0=Empty, 1=White, 2=Black

            piece_char = ID_TO_PIECE[pred_label]
            expected_color = 0
            if piece_char != 'e':
                expected_color = 1 if piece_char.isupper() else 2

            is_ood = pred_color_label != expected_color
            if centroids is not None:
                embedding = F.normalize(embedding, p=2, dim=1)
                dists = torch.cdist(embedding, centroids, p=2)
                min_dist, _ = torch.min(dists, dim=1)
                if min_dist.item() > ood_threshold:
                    is_ood = True

            return ID_TO_PIECE[pred_label], is_ood

        if model_type == "ResNetWithEmbeddings":
            logits, embedding = model(tile_tensor)
            probs = F.softmax(logits, dim=1)
            
            # Hybrid Approach:
            # 1. Classification based on Softmax Probabilities (usually more accurate for class ID)
            conf, pred_idx = torch.max(probs, 1)
            pred_label = pred_idx.item()
            
            # 2. OOD Detection based on Embedding Distance (better for anomaly detection)
            is_ood = False
            if centroids is not None:
                embedding = F.normalize(embedding, p=2, dim=1)
                dists = torch.cdist(embedding, centroids, p=2)
                min_dist, _ = torch.min(dists, dim=1)
                
                if min_dist.item() > ood_threshold:
                    is_ood = True
            else:
                # Fallback if no centroids
                if conf.item() < 0.6:
                    is_ood = True

            return ID_TO_PIECE[pred_label], is_ood

        else: # PieceClassifier
            logits = model(tile_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            pred_label = pred_idx.item()
            
            # Heuristic OOD: If confidence is low, consider it OOD/Uncertain
            # Since PieceClassifier doesn't have embeddings, we use probability
            is_ood = conf.item() < 0.6 
            
            return ID_TO_PIECE.get(pred_label, 'e'), is_ood

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Chess Board Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory containing models")
    parser.add_argument("--model", type=str, default=None, help="Specific model filename to load (optional)")
    parser.add_argument("--ood_threshold", type=float, default=0.8, help="Threshold for OOD detection (distance or confidence)")
    args = parser.parse_args()
    
    # 1. Find Model
    if not os.path.exists(args.checkpoints_dir):
        print(f"Error: Checkpoints directory '{args.checkpoints_dir}' not found.")
        return

    if args.model:
        model_path = os.path.join(args.checkpoints_dir, args.model)
        if not os.path.exists(model_path):
             print(f"Error: Model file '{model_path}' not found.")
             return
        print(f"Loading specific model from {model_path}")
    else:
        best_path = os.path.join(args.checkpoints_dir, "resnet18_best.pth")
        if os.path.exists(best_path):
            model_path = best_path
            print(f"Loading best model from {model_path}")
        else:
            epoch, model_path = get_latest_epoch_model(args.checkpoints_dir)
            if not model_path:
                print(f"No model_epoch_*.pth or resnet18_best.pth files found in {args.checkpoints_dir}")
                return
            print(f"Loading latest epoch model from {model_path} (Epoch {epoch})")
    
    # 2. Load Model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect Architecture based on state_dict keys
    keys = list(checkpoint.keys())
    if any(k.startswith('backbone') for k in keys):
        if 'fc_color.weight' in keys:
            model_type = "ResNetMultiHead"
            model = ResNetMultiHead(num_piece_classes=13, num_color_classes=3).to(device)
        else:
            model_type = "ResNetWithEmbeddings"
            model = ResNetWithEmbeddings(num_classes=13).to(device)
    else:
        model_type = "PieceClassifier"
        model = PieceClassifier(num_classes=13).to(device)
        
    model.load_state_dict(checkpoint, strict=False) # strict=False to be safe with minor version diffs
    model.eval()
    print(f"Detected architecture: {model_type}")
    
    # Load Centroids if ResNet
    centroids = None
    if model_type in ("ResNetWithEmbeddings", "ResNetMultiHead"):
        centroids_path = os.path.join(args.checkpoints_dir, "centroids.pt")
        if os.path.exists(centroids_path):
            centroids = torch.load(centroids_path, map_location=device)
            print(f"Loaded centroids from {centroids_path}")
        else:
            print("Warning: centroids.pt not found. Using confidence fallback for OOD.")

    # 3. Process Image
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        return

    img = Image.open(args.image).convert("RGB")
    original_img = img.copy()
    
    # Resize to 640x640 for cutting (standard from training)
    img_resized = img.resize((640, 640), resample=Image.BILINEAR)
    
    # Transforms
    to_tensor = transforms.ToTensor()
    resnet_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    board_grid = [] # 8x8 chars
    ood_mask = []   # List of (row, col) that are OOD
    
    # Tile size is 80, but we crop 160x160 (padding 40px)
    TILE_SIZE = 80
    PADDING_PX = 40
    FINAL_SIZE = 160
    BOARD_SIZE = 640
    
    for r in range(8):
        row_pieces = []
        for c in range(8):
            # Calculate base tile coordinates
            base_left = c * TILE_SIZE
            base_top = r * TILE_SIZE
            base_right = base_left + TILE_SIZE
            base_bottom = base_top + TILE_SIZE
            
            # Calculate padded coordinates
            pad_left = base_left - PADDING_PX
            pad_top = base_top - PADDING_PX
            pad_right = base_right + PADDING_PX
            pad_bottom = base_bottom + PADDING_PX
            
            # Handle boundaries
            crop_left = max(0, pad_left)
            crop_top = max(0, pad_top)
            crop_right = min(BOARD_SIZE, pad_right)
            crop_bottom = min(BOARD_SIZE, pad_bottom)
            
            tile_img = img_resized.crop((crop_left, crop_top, crop_right, crop_bottom))
            
            # Pad with black if needed
            if tile_img.size != (FINAL_SIZE, FINAL_SIZE):
                new_tile = Image.new("RGB", (FINAL_SIZE, FINAL_SIZE), (0, 0, 0))
                paste_x = abs(pad_left) if pad_left < 0 else 0
                paste_y = abs(pad_top) if pad_top < 0 else 0
                new_tile.paste(tile_img, (paste_x, paste_y))
                tile_img = new_tile
            
            # Prepare tensor
            if model_type in ("ResNetWithEmbeddings", "ResNetMultiHead"):
                input_tensor = resnet_transform(tile_img)
            else:
                input_tensor = to_tensor(tile_img)
                
            piece, is_ood = infer_tile(model, input_tensor, device, model_type, centroids, args.ood_threshold)
            row_pieces.append(piece)
            
            if is_ood:
                ood_mask.append((r, c))
                
        board_grid.append(row_pieces)
        
    # 4. Reconstruct FEN
    fen = fen_from_board(board_grid)
    print(f"\nPredicted FEN: {fen}")
    
    # 5. Generate Offline FEN Image using chessboard-image
    output_filename = "inference_result.png"
    
    # Generate the base board image from FEN
    # cbi.generate_image returns True on success, saves to path
    print(f"Generating FEN diagram using chessboard-image...")
    cbi.generate_image(fen, output_filename, size=480, show_coordinates=True)
    
    # 6. Overlay OOD markers
    # Open the generated image to draw on it
    if os.path.exists(output_filename):
        fen_img = Image.open(output_filename).convert("RGB")
        draw = ImageDraw.Draw(fen_img)
        w, h = fen_img.size
        
        # Calculate board area (might be different due to coordinates)
        # chessboard-image usually adds border for coordinates.
        # Let's assume standard square board behavior for now or check size.
        # If coordinates are shown, the actual board is smaller.
        # But we asked for size=480. Let's see if we can overlay accurately.
        # A safer bet for exact overlay is to use the package's internal drawing or just standard calculation
        # assuming the image is mostly the board.
        
        # With coordinates, there is a margin.
        # chessboard-image centers the board.
        # Let's do a simple full-image grid if coordinates are disabled or try to detect.
        # For robustness, let's disable coordinates for the OOD overlay version to ensure grid alignment
        # Or re-generate without coordinates for the overlay logic.
        
        cbi.generate_image(fen, output_filename, size=480, show_coordinates=False)
        fen_img = Image.open(output_filename).convert("RGB")
        draw = ImageDraw.Draw(fen_img)
        w, h = fen_img.size
        
        cell_w = w / 8
        cell_h = h / 8
    
        for r, c in ood_mask:
            x_min = c * cell_w
            y_min = r * cell_h
            x_max = (c + 1) * cell_w
            y_max = (r + 1) * cell_h
            
            margin_x = cell_w * 0.2
            margin_y = cell_h * 0.2
            
            draw.line(
                [(x_min + margin_x, y_min + margin_y), (x_max - margin_x, y_max - margin_y)], 
                fill="red", width=5
            )
            draw.line(
                [(x_min + margin_x, y_max - margin_y), (x_max - margin_x, y_min + margin_y)], 
                fill="red", width=5
            )
            
        fen_img.save(output_filename)
        print(f"Saved generated FEN diagram to {output_filename}")
    
    # Removed overlay on original image as requested

if __name__ == "__main__":
    main()
