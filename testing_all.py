import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
import chessboard_image as cbi
from tqdm import tqdm
import importlib.util
import sys
import glob
from chess_model_protocol import Piece
import inspect

def load_model_module(model_file_path):
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)
    
    # Find the class that implements ChessModelProtocol
    # We need to find a class that inherits from BaseChessModel (or ChessModelProtocol)
    # AND is NOT BaseChessModel itself (which is abstract).
    
    found_class = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type):
            # Check if it inherits from ChessModelProtocol (directly or indirectly)
            bases = [b.__name__ for b in obj.__mro__]
            if "ChessModelProtocol" in bases:
                # Skip the abstract base classes themselves
                if name in ["ChessModelProtocol", "BaseChessModel"]:
                    continue
                
                # CRITICAL FIX: Only pick classes defined IN THIS MODULE
                # This prevents picking up BaseChessModel which is imported
                if obj.__module__ != "model_module":
                    continue
                
                # Ensure it's not abstract
                if inspect.isabstract(obj):
                    continue

                found_class = obj
                break
    
    if found_class:
        return found_class()
    else:
        # Fallback: print what was found to help debug
        print(f"Debug: Classes found in {model_file_path}:")
        for name, obj in module.__dict__.items():
            if isinstance(obj, type):
                print(f" - {name} (Module: {obj.__module__})")
        raise ValueError("No concrete class implementing ChessModelProtocol found in the provided file.")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="unlalbled")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the python file implementing the model protocol")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--ood_threshold", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model_protocol = load_model_module(args.model_file)
    model = model_protocol.create_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Standard transform for inference
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
                    
                    piece_enum = model_protocol.infer_tile(model, input_tensor, device, args.ood_threshold)
                    
                    if piece_enum == Piece.OOD:
                        row_pieces.append('e') # Treat OOD as empty for FEN generation, but mark it
                        ood_mask.append((r, c))
                    else:
                        row_pieces.append(piece_enum.value)

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
