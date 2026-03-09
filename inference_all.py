import os
import argparse
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
import chessboard_image as cbi
import importlib.util
import sys
import inspect
from chess_model_protocol import Piece, ChessModelProtocol


def load_model_module(model_file_path):
    spec = importlib.util.spec_from_file_location("model_module", model_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = module
    spec.loader.exec_module(module)

    # Safely find the concrete class implementing ChessModelProtocol
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ChessModelProtocol) and not inspect.isabstract(obj):
            return obj()

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
    parser = argparse.ArgumentParser(description="Run inference on a single chess image.")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the python file implementing the model protocol (e.g., model_v3.py)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the result image. Defaults to 'result_<filename>' in the same directory.")
    parser.add_argument("--ood_threshold", type=float, default=0.7, help="Confidence threshold for OOD detection")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model Protocol
    try:
        model_protocol = load_model_module(args.model_file)
    except Exception as e:
        print(f"Error loading model protocol: {e}")
        return

    # Initialize Model
    model = model_protocol.create_model().to(device)

    # Load Weights
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Prepare Image
    try:
        img = Image.open(args.image_path).convert("RGB")
        original_img = img.copy()

        # Resize to standard 480x480 board size
        img_resized = img.resize((480, 480), resample=Image.BILINEAR)

        # CRITICAL FIX: Add the 18-pixel black padding to replicate process_chess_final.py
        pad = 18
        img_padded = ImageOps.expand(img_resized, border=pad, fill='black')

    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Inference Transform
    infer_transform = transforms.Compose([
        transforms.Resize((96, 96)),  # Failsafe, though the crop is already 96x96
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    board_grid = []
    ood_mask = []

    # Use exact dimensions from process_chess_final.py
    square_size = 60
    tile_size = 96

    print("Running inference...")
    for r in range(8):
        row_pieces = []
        for c in range(8):
            # Because the image is padded by 18px, this perfectly centers the 96x96 window
            # over the 60x60 logical square, resulting in exactly a 1.6x expansion.
            y_start = r * square_size
            y_end = y_start + tile_size
            x_start = c * square_size
            x_end = x_start + tile_size

            # Crop the exact 96x96 tile from the padded image
            tile_img = img_padded.crop((x_start, y_start, x_end, y_end))
            input_tensor = infer_transform(tile_img)

            # Use Protocol for Inference
            piece_enum = model_protocol.infer_tile(model, input_tensor, device, args.ood_threshold)

            if piece_enum == Piece.OOD:
                row_pieces.append('e')  # Treat OOD as empty for FEN generation
                ood_mask.append((r, c))
            else:
                row_pieces.append(piece_enum.value)

        board_grid.append(row_pieces)

    # Generate FEN
    fen = fen_from_board(board_grid)
    print(f"Predicted FEN: {fen}")

    # Visualization
    try:
        if args.output_path:
            out_path = args.output_path
        else:
            dir_name = os.path.dirname(args.image_path)
            base_name = os.path.basename(args.image_path)
            out_path = os.path.join(dir_name, f"result_{base_name}")

        temp_fen_path = "temp_fen_inference.png"
        cbi.generate_image(fen, temp_fen_path, size=480, show_coordinates=False)

        if os.path.exists(temp_fen_path):
            fen_img = Image.open(temp_fen_path).convert("RGB")
            draw = ImageDraw.Draw(fen_img)
            cell_w, cell_h = fen_img.width / 8, fen_img.height / 8

            # Draw OOD markers
            for r, c in ood_mask:
                x_min, y_min = c * cell_w, r * cell_h
                x_max, y_max = (c + 1) * cell_w, (r + 1) * cell_h
                m_x, m_y = cell_w * 0.2, cell_h * 0.2
                draw.line([(x_min + m_x, y_min + m_y), (x_max - m_x, y_max - m_y)], fill="red", width=5)
                draw.line([(x_min + m_x, y_max - m_y), (x_max - m_x, y_min + m_y)], fill="red", width=5)

            # Combine Images
            display_img = original_img.resize((480, 480))
            combined = Image.new('RGB', (display_img.width + fen_img.width, 480), (255, 255, 255))
            combined.paste(display_img, (0, 0))
            combined.paste(fen_img, (display_img.width, 0))

            combined.save(out_path)
            print(f"Result saved to: {out_path}")

            os.remove(temp_fen_path)
        else:
            print("Error: Failed to generate FEN image.")

    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()