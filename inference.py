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
        if model_type == "ResNetWithEmbeddings":
            logits, embedding = model(tile_tensor)
            probs = F.softmax(logits, dim=1)
            
            # OOD Check using Centroids if available
            is_ood = False
            if centroids is not None:
                embedding = F.normalize(embedding, p=2, dim=1)
                # Compute distances (1, 13)
                dists = torch.cdist(embedding, centroids, p=2)
                min_dist, pred_idx = torch.min(dists, dim=1)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Chess Board Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_resnet_triplet", help="Directory containing models")
    parser.add_argument("--ood_threshold", type=float, default=0.8, help="Threshold for OOD detection (distance or confidence)")
    args = parser.parse_args()
    
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
    
    # Resize to 640x640 for cutting (standard from training)
    img_resized = img.resize((640, 640), resample=Image.BILINEAR)
    
    # Transforms
    to_tensor = transforms.ToTensor()
    resnet_transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    board_grid = [] # 8x8 chars
    ood_mask = []   # List of (row, col) that are OOD
    
    tile_size = 80
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
    
    # 5. Generate Offline FEN Image using chessboard-image
    output_filename = "inference_result.png"
    
    # Generate the base board image from FEN
    # cbi.generate_image returns True on success, saves to path
    print(f"Generating FEN diagram using chessboard-image...")
    cbi.generate_image(
        fen=fen,
        output_path=output_filename,
        size=480,
        show_coordinates=True
    )
    
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
        
        cbi.generate_image(
            fen=fen,
            output_path=output_filename,
            size=480,
            show_coordinates=False
        )
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
    
    # Save overlay as well
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
        
        draw.line(
            [(x_min + margin_x, y_min + margin_y), (x_max - margin_x, y_max - margin_y)], 
            fill="red", width=5
        )
        draw.line(
            [(x_min + margin_x, y_max - margin_y), (x_max - margin_x, y_min + margin_y)], 
            fill="red", width=5
        )
        
    original_img.save(overlay_filename)
    print(f"Saved original overlay to {overlay_filename}")

if __name__ == "__main__":
    main()
