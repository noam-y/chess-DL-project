import os
import glob
import argparse
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import chessboard_image as cbi
import numpy as np
from tqdm import tqdm
import re

# Import models
try:
    from train import PieceClassifier
except ImportError:
    pass

try:
    from nir_train import ResNetWithEmbeddings, ID_TO_PIECE
except ImportError:
    # Fallback
    PIECE_TO_ID = {
        'e': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
    }
    ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

def get_latest_epoch_model(checkpoints_dir):
    files = glob.glob(os.path.join(checkpoints_dir, "*_epoch_*.pth"))
    if not files:
        return None, None
    
    epoch_files = []
    for f in files:
        match = re.search(r"_epoch_(\d+)\.pth", f)
        if match:
            epoch_files.append((int(match.group(1)), f))
    
    if not epoch_files:
        return None, None
        
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    return epoch_files[0]

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

def infer_tile(model, tile_tensor, device, model_type, centroids=None, ood_threshold=1.0):
    tile_tensor = tile_tensor.to(device).unsqueeze(0)
    
    with torch.no_grad():
        if model_type == "ResNetWithEmbeddings":
            logits, embedding = model(tile_tensor)
            probs = F.softmax(logits, dim=1)
            
            # Hybrid Approach:
            # 1. Classification based on Softmax Probabilities
            conf, pred_idx = torch.max(probs, 1)
            pred_label = pred_idx.item()
            
            # 2. OOD Detection based on Embedding Distance
            is_ood = False
            dist_val = 0.0
            if centroids is not None:
                embedding = F.normalize(embedding, p=2, dim=1)
                dists = torch.cdist(embedding, centroids, p=2)
                min_dist, _ = torch.min(dists, dim=1)
                dist_val = min_dist.item()
                
                if dist_val > ood_threshold:
                    is_ood = True
            else:
                # Fallback if no centroids
                if conf.item() < 0.6:
                    is_ood = True

            return ID_TO_PIECE[pred_label], is_ood, dist_val

        else:
            logits = model(tile_tensor)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            pred_label = pred_idx.item()
            is_ood = conf.item() < 0.6 
            return ID_TO_PIECE.get(pred_label, 'e'), is_ood, 0.0

def main():
    parser = argparse.ArgumentParser(description="Batch Inference on Unlabeled Images")
    parser.add_argument("--input_dir", type=str, default="unlalbled", help="Directory with input images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_resnet_multihead", help="Directory containing models")
    parser.add_argument("--model", type=str, default=None, help="Specific model filename to load (optional)")
    parser.add_argument("--ood_threshold", type=float, default=0.8, help="Threshold for OOD detection")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load Model
    if args.model:
        model_path = os.path.join(args.checkpoints_dir, args.model)
        if not os.path.exists(model_path):
             print(f"Error: Model file '{model_path}' not found.")
             return
        print(f"Loading specific model from {model_path}")
        epoch = 0 
    else:
        # Try finding 'resnet18_best.pth' first
        best_path = os.path.join(args.checkpoints_dir, "resnet18_best.pth")
        if os.path.exists(best_path):
            model_path = best_path
            epoch = 0 # unknown
            print(f"Loading best model from {model_path}")
        else:
             epoch, model_path = get_latest_epoch_model(args.checkpoints_dir)
    
    if not model_path:
        print(f"No model found in {args.checkpoints_dir}")
        return

    print(f"Loading model from {model_path} (Epoch {epoch})")
    checkpoint = torch.load(model_path, map_location=device)
    
    keys = list(checkpoint.keys())
    if any(k.startswith('backbone') for k in keys):
        model_type = "ResNetWithEmbeddings"
        model = ResNetWithEmbeddings(num_classes=13).to(device)
    else:
        model_type = "PieceClassifier"
        model = PieceClassifier(num_classes=13).to(device)
        
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    centroids = None
    if model_type == "ResNetWithEmbeddings":
        centroids_path = os.path.join(os.path.dirname(model_path), "centroids.pt")
        if os.path.exists(centroids_path):
            centroids = torch.load(centroids_path, map_location=device)
            print(f"Loaded centroids from {centroids_path}")

    # Transforms
    to_tensor = transforms.ToTensor()
    resnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process Images
    image_files = glob.glob(os.path.join(args.input_dir, "*.jpg")) + glob.glob(os.path.join(args.input_dir, "*.png"))
    print(f"Found {len(image_files)} images in {args.input_dir}")

    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            original_img = img.copy()
            
            # Resize for cutting
            img_resized = img.resize((480, 480), resample=Image.BILINEAR)
            
            board_grid = []
            ood_mask = []
            tile_size = 60
            
            for r in range(8):
                row_pieces = []
                for c in range(8):
                    left = c * tile_size
                    upper = r * tile_size
                    tile_img = img_resized.crop((left, upper, left+tile_size, upper+tile_size))
                    
                    if model_type == "ResNetWithEmbeddings":
                        input_tensor = resnet_transform(tile_img)
                    else:
                        input_tensor = to_tensor(tile_img)
                        
                    piece, is_ood, dist_val = infer_tile(model, input_tensor, device, model_type, centroids, args.ood_threshold)
                    row_pieces.append(piece)
                    if is_ood:
                        ood_mask.append((r, c))
                        # Optional: Print distance for debugging
                        # print(f"OOD at ({r},{c}): {piece} dist={dist_val:.4f}")
                board_grid.append(row_pieces)
            
            fen = fen_from_board(board_grid)
            
            # Generate FEN Image
            fen_img_path = os.path.join(args.output_dir, "temp_fen.png")
            cbi.generate_image(
                fen_str=fen,
                output_path=fen_img_path,
                size=480,
                show_coordinates=True 
            )
            
            # Generate OOD Overlay on FEN Image (No coordinates for alignment)
            ood_fen_path = os.path.join(args.output_dir, "temp_ood_fen.png")
            cbi.generate_image(
                fen_str=fen,
                output_path=ood_fen_path,
                size=480,
                show_coordinates=False
            )
            
            if os.path.exists(fen_img_path) and os.path.exists(ood_fen_path):
                fen_img = Image.open(fen_img_path).convert("RGB")
                ood_fen_img = Image.open(ood_fen_path).convert("RGB")
                
                # Draw OOD on the clean FEN image (ood_fen_img)
                draw = ImageDraw.Draw(ood_fen_img)
                w, h = ood_fen_img.size
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
                
                # Now we want the final right-side image to be the FEN diagram with OOD markers.
                # But wait, if we use the one without coordinates for drawing, it looks different than the one with coordinates.
                # Let's stick to the user's request: "inference_result.png" style.
                # In inference.py we generate one with coordinates, then overwrite it with one without coordinates for the OOD drawing.
                # Let's just use the one without coordinates for the FEN side to ensure OOD alignment is easy.
                
                final_fen_img = ood_fen_img # This has the OOD markers drawn on it
                
                # Create Side-by-Side
                # Left: Original Image
                # Right: FEN Diagram with OOD markers
                
                # Resize original to match height of FEN image
                display_img = original_img.resize((final_fen_img.height, final_fen_img.height))
                
                total_width = display_img.width + final_fen_img.width
                max_height = max(display_img.height, final_fen_img.height)
                
                combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                combined.paste(display_img, (0, 0))
                combined.paste(final_fen_img, (display_img.width, 0))
                
                base_name = os.path.basename(img_path)
                save_path = os.path.join(args.output_dir, f"result_{base_name}")
                combined.save(save_path)
                
                # Cleanup temp
                if os.path.exists(fen_img_path): os.remove(fen_img_path)
                if os.path.exists(ood_fen_path): os.remove(ood_fen_path)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
