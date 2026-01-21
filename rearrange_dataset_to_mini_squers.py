import os
import glob
import pandas as pd
from PIL import Image
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
    IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
    GT_FILE = os.path.join(OUTPUT_DIR, 'gt.csv')
    
    # Create directories
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        print(f"Created directory: {IMAGES_DIR}")
    
    gt_data = []
    image_counter = 1
    
    # Find all CSV files recursively
    csv_files = glob.glob(os.path.join(BASE_DIR, '**', '*.csv'), recursive=True)
    print(f"Found {len(csv_files)} CSV files in {BASE_DIR}")
    
    for csv_path in csv_files:
        print(f"Processing {csv_path}...")
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
            for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(csv_path)):
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
                    img = img.resize((640, 640), resample=Image.BILINEAR)
                    
                    # Parse FEN into 8x8 grid
                    grid = parse_fen_to_grid(fen)
                    
                    # Crop into 64 tiles
                    tile_size = 80
                    for r in range(8):
                        for c in range(8):
                            left = c * tile_size
                            upper = r * tile_size
                            right = left + tile_size
                            lower = upper + tile_size
                            
                            tile = img.crop((left, upper, right, lower))
                            
                            # Save tile
                            tile_filename = f"{image_counter}.jpg"
                            tile_path = os.path.join(IMAGES_DIR, tile_filename)
                            tile.save(tile_path, quality=95)
                            
                            # Add to ground truth
                            label = grid[r][c]
                            gt_data.append({'image_name': tile_filename, 'piece_id': label})
                            
                            image_counter += 1
                            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Ground Truth CSV
    print(f"Saving ground truth to {GT_FILE}...")
    gt_df = pd.DataFrame(gt_data)
    # Write header as implied by prompt structure description
    gt_df.to_csv(GT_FILE, index=False)
    
    print(f"Done! Processed {len(gt_data)} tiles.")

if __name__ == "__main__":
    main()
