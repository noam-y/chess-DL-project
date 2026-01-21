import os
import glob
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

def parse_fen_to_grid(fen):
    board = []
    # Take the board part of the FEN (everything before the first space)
    fen_board = fen.split(' ')[0]
    rows = fen_board.split('/')
    for row_str in rows:
        row = []
        for char in row_str:
            if char.isdigit():
                # Empty squares
                row.extend(['e'] * int(char))
            else:
                # Piece
                row.append(char)
        board.append(row)
    return board

def main():
    BASE_DIR = 'assets/labeled_data'
    OUTPUT_DIR = 'assets/dataset'
    
    TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
    TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
    
    # Create directories
    for d in [TRAIN_DIR, TEST_DIR]:
        images_dir = os.path.join(d, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            print(f"Created directory: {images_dir}")
    
    # Configuration
    BOARD_SIZE = 640  # 8 * 80
    TILE_SIZE = 80
    PADDING_RATIO = 0.5 # 50% padding on each side
    PADDING_PX = int(TILE_SIZE * PADDING_RATIO) # 40 pixels
    FINAL_SIZE = TILE_SIZE + 2 * PADDING_PX 
    
    print(f"Generating tiles of size {FINAL_SIZE}x{FINAL_SIZE} (Tile {TILE_SIZE} + Padding {PADDING_PX} per side)")
    
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(BASE_DIR, '**', '*.csv'), recursive=True)
    print(f"Found {len(csv_files)} CSV files in {BASE_DIR}")
    
    train_games = ['game2', 'game4', 'game6']
    # All other games go to test
    
    train_data = []
    test_data = []
    
    image_counter_train = 1
    image_counter_test = 1
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        game_name = filename.split('.')[0] # e.g. "game2.csv" -> "game2"
        
        # Determine split
        if any(g in game_name for g in train_games):
            split = 'train'
            current_data = train_data
            current_counter = image_counter_train
            base_output_dir = TRAIN_DIR
        else:
            split = 'test'
            current_data = test_data
            current_counter = image_counter_test
            base_output_dir = TEST_DIR
            
        print(f"Processing {csv_path} -> {split}...")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Determine image directory
            game_folder = os.path.dirname(csv_path)
            images_base_path = os.path.join(game_folder, 'tagged_images')
            
            if not os.path.exists(images_base_path):
                print(f"Warning: tagged_images directory not found at {images_base_path}, skipping...")
                continue
                
            # Process each frame
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{game_name} ({split})"):
                # Handle potentially missing columns
                if 'from_frame' not in df.columns or 'fen' not in df.columns:
                    continue
                    
                frame_id = row['from_frame']
                fen = row['fen']
                
                # Construct image filename
                img_name = f"frame_{int(frame_id):06d}.jpg"
                img_path = os.path.join(images_base_path, img_name)
                
                if not os.path.exists(img_path):
                    continue
                    
                # Load and process image
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    # Resize to 640x640 to get 80x80 squares (8 * 80 = 640)
                    img = img.resize((BOARD_SIZE, BOARD_SIZE), resample=Image.BILINEAR)
                    
                    # Parse FEN into 8x8 grid
                    grid = parse_fen_to_grid(fen)
                    
                    # Crop into 64 tiles with padding
                    for r in range(8):
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
                            
                            tile = img.crop((crop_left, crop_top, crop_right, crop_bottom))
                            
                            # Pad with black if needed
                            if tile.size != (FINAL_SIZE, FINAL_SIZE):
                                new_tile = Image.new("RGB", (FINAL_SIZE, FINAL_SIZE), (0, 0, 0))
                                paste_x = abs(pad_left) if pad_left < 0 else 0
                                paste_y = abs(pad_top) if pad_top < 0 else 0
                                new_tile.paste(tile, (paste_x, paste_y))
                                tile = new_tile
                            
                            # Save tile
                            tile_filename = f"{current_counter}.jpg"
                            tile_path = os.path.join(base_output_dir, 'images', tile_filename)
                            tile.save(tile_path, quality=95)
                            
                            # Add to ground truth
                            label = grid[r][c]
                            current_data.append({'image_name': tile_filename, 'piece_id': label})
                            
                            current_counter += 1
        
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            
        # Update global counters
        if split == 'train':
            image_counter_train = current_counter
        else:
            image_counter_test = current_counter
            
    # Save Ground Truth CSVs
    print("Saving ground truth CSVs...")
    pd.DataFrame(train_data).to_csv(os.path.join(TRAIN_DIR, 'gt.csv'), index=False)
    pd.DataFrame(test_data).to_csv(os.path.join(TEST_DIR, 'gt.csv'), index=False)
    
    print(f"Done! Train tiles: {len(train_data)}, Test tiles: {len(test_data)}")

if __name__ == "__main__":
    main()
