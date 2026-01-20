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

# Import models from their respective files to avoid duplication
try:
    from train import PieceClassifier
except ImportError:
    print("Warning: Could not import PieceClassifier from train.py")

try:
    from nir_train import ResNetWithEmbeddings, ID_TO_PIECE
except ImportError:
    print("Warning: Could not import ResNetWithEmbeddings from nir_train.py")
    # Fallback ID mapping if import fails
    PIECE_TO_ID = {
        'e': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

# --- Utils ---

def get_latest_epoch_model(checkpoints_dir):
    # Pattern: model_epoch_{epoch}.pth
    files = glob.glob(os.path.join(checkpoints_dir, "model_epoch_*.pth"))
    if not files:
        return None, None
    
    # Extract epochs
    epoch_files = []
    for f in files:
        match = re.search(r"model_epoch_(\d+)\.pth", f)
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

def create_visual_board(board_grid, ood_mask):
    # Setup colors
    colors = ["#F0D9B5", "#B58863"] # Light, Dark (Standard Chess)
    
    width, height = 480, 480
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    font = None
    try:
        # Common paths for Linux/Mac
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/freefont/FreeSansBold.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/Library/Fonts/Arial.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 40)
                break
        if font is None:
             font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    tile_size = 60
    
    for r in range(8):
        for c in range(8):
            x = c * tile_size
            y = r * tile_size
            
            # Draw background square
            color_idx = (r + c) % 2
            draw.rectangle([x, y, x+tile_size, y+tile_size], fill=colors[color_idx])
            
            # Draw Piece
            piece_char = board_grid[r][c]
            if piece_char != 'e':
                # Determine color
                if piece_char.isupper(): # White piece
                    text_color = "white"
                    outline_color = "black"
                else: # Black piece
                    text_color = "black"
                    outline_color = "white"
                
                # Draw text with outline for visibility
                # Note: stroke_width available in Pillow >= 6.2.0
                try:
                    draw.text((x+15, y+10), piece_char, font=font, fill=text_color, stroke_width=2, stroke_fill=outline_color)
                except TypeError:
                    # Fallback for older Pillow
                    draw.text((x+15, y+10), piece_char, font=font, fill=text_color)
            
            # Draw OOD X
            if (r, c) in ood_mask:
                 draw.line([x, y, x+tile_size, y+tile_size], fill="red", width=4)
                 draw.line([x, y+tile_size, x+tile_size, y], fill="red", width=4)

    return img

def infer_tile(model, tile_tensor, device, model_type, centroids=None, ood_threshold=1.0):
    tile_tensor = tile_tensor.to(device).unsqueeze(0) # (1, 3, H, W)
    
    with torch.no_grad():
        if model_type == "ResNetWithEmbeddings":
            logits, embedding = model(tile_tensor)
            probs = F.softmax(logits, dim=1)
            
            # OOD Check using Centroids if available
            is_ood = False
            if centroids is not None:
                embedding = F.normalize(embedding, p=2, dim=1)
                dists = torch.cdist(embedding, centroids.unsqueeze(0), p=2).squeeze(0)
                min_dist, pred_idx = torch.min(dists, dim=0)
                if min_dist.item() > ood_threshold:
                    is_ood = True
                pred_label = pred_idx.item()
            else:
                # Fallback to logits
                conf, pred_idx = torch.max(probs, 1)
                pred_label = pred_idx.item()
                if conf.item() < 0.5: # arbitrary low confidence threshold
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
    parser = argparse.ArgumentParser(description="Chess Board Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory containing models")
    parser.add_argument("--ood_threshold", type=float, default=0.8, help="Threshold for OOD detection (distance or confidence)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Find Model
    if not os.path.exists(args.checkpoints_dir):
        print(f"Error: Checkpoints directory '{args.checkpoints_dir}' not found.")
        return

    epoch, model_path = get_latest_epoch_model(args.checkpoints_dir)
    if not model_path:
        print(f"No model_epoch_*.pth files found in {args.checkpoints_dir}")
        return
        
    print(f"Loading model from {model_path} (Epoch {epoch})")
    
    # 2. Load Model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect Architecture based on state_dict keys
    keys = list(checkpoint.keys())
    if any(k.startswith('backbone') for k in keys):
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
    if model_type == "ResNetWithEmbeddings":
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
    
    # Resize to 480x480 for cutting (standard from training)
    img_resized = img.resize((480, 480), resample=Image.BILINEAR)
    
    # Transforms
    to_tensor = transforms.ToTensor()
    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    board_grid = [] # 8x8 chars
    ood_mask = []   # List of (row, col) that are OOD
    
    tile_size = 60
    for r in range(8):
        row_pieces = []
        for c in range(8):
            left = c * tile_size
            upper = r * tile_size
            tile_img = img_resized.crop((left, upper, left+tile_size, upper+tile_size))
            
            # Prepare tensor
            if model_type == "ResNetWithEmbeddings":
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
    
    # 5. Create Visual FEN Diagram
    fen_img = create_visual_board(board_grid, ood_mask)
    output_filename = "inference_result.jpg"
    fen_img.save(output_filename)
    print(f"Saved visual FEN diagram to {output_filename}")
    
    # Optional: Also save the overlay on original image
    overlay_filename = "inference_overlay.jpg"
    draw = ImageDraw.Draw(original_img)
    w, h = original_img.size
    cell_w = w / 8
    cell_h = h / 8
    
    for r, c in ood_mask:
        x_min = c * cell_w
        y_min = r * cell_h
        x_max = (c + 1) * cell_w
        y_max = (r + 1) * cell_h
        margin_x = cell_w * 0.2
        margin_y = cell_h * 0.2
        draw.line([(x_min + margin_x, y_min + margin_y), (x_max - margin_x, y_max - margin_y)], fill="red", width=5)
        draw.line([(x_min + margin_x, y_max - margin_y), (x_max - margin_x, y_min + margin_y)], fill="red", width=5)
    
    original_img.save(overlay_filename)
    print(f"Saved original overlay to {overlay_filename}")

if __name__ == "__main__":
    main()
