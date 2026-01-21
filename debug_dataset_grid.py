import os
import glob
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def parse_fen_to_grid(fen):
    board = []
    fen_board = fen.split(' ')[0]
    rows = fen_board.split('/')
    for row_str in rows:
        row = []
        for char in row_str:
            if char.isdigit():
                row.extend(['e'] * int(char))
            else:
                row.append(char)
        board.append(row)
    return board

def main():
    BASE_DIR = 'assets/labeled_data'
    DEBUG_DIR = 'debug_grid_output'
    
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)
        print(f"Created debug directory: {DEBUG_DIR}")
    
    # Get just one CSV to test
    csv_files = glob.glob(os.path.join(BASE_DIR, '**', '*.csv'), recursive=True)
    if not csv_files:
        print("No CSV files found!")
        return

    target_csv = csv_files[0] # Pick the first one
    print(f"Debugging with: {target_csv}")
    
    df = pd.read_csv(target_csv)
    df.columns = df.columns.str.strip()
    
    game_folder = os.path.dirname(target_csv)
    images_base_path = os.path.join(game_folder, 'tagged_images')
    
    # Process just the first 5 frames
    for i, row in df.head(5).iterrows():
        frame_id = row['from_frame']
        fen = row['fen']
        
        img_name = f"frame_{int(frame_id):06d}.jpg"
        img_path = os.path.join(images_base_path, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        print(f"Processing {img_name}...")
        
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            # Resize to 640x640 (current logic)
            img_resized = img.resize((640, 640), resample=Image.BILINEAR)
            
            # Create a copy to draw grid on
            debug_img = img_resized.copy()
            draw = ImageDraw.Draw(debug_img)
            
            grid = parse_fen_to_grid(fen)
            tile_size = 80
            
            # Draw Grid lines
            for x in range(0, 641, tile_size):
                draw.line([(x, 0), (x, 640)], fill="red", width=2)
            for y in range(0, 641, tile_size):
                draw.line([(0, y), (640, y)], fill="red", width=2)
                
            # Draw Labels and Save Tiles
            for r in range(8):
                for c in range(8):
                    label = grid[r][c]
                    
                    left = c * tile_size
                    upper = r * tile_size
                    right = left + tile_size
                    lower = upper + tile_size
                    
                    # Draw label on the full debug image
                    # Calculate center
                    cx = left + tile_size // 2
                    cy = upper + tile_size // 2
                    draw.text((cx-5, cy-5), label, fill="yellow") # Simple text
                    
                    # Save individual tile to check crop quality
                    tile = img_resized.crop((left, upper, right, lower))
                    tile_filename = f"frame_{int(frame_id)}_tile_{r}_{c}_label_{label}.jpg"
                    tile.save(os.path.join(DEBUG_DIR, tile_filename))

            # Save the full debug grid image
            debug_img.save(os.path.join(DEBUG_DIR, f"debug_grid_frame_{int(frame_id)}.jpg"))

    print(f"Debug output saved to {DEBUG_DIR}")
    print("Please inspect 'debug_grid_frame_*.jpg' to see if the red lines align with the board squares.")
    print("Please inspect individual tile images to see if they contain the correct piece centered.")

if __name__ == "__main__":
    main()
