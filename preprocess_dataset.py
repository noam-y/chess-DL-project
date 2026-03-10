import cv2
import pandas as pd
from pathlib import Path


def fen_to_grid(fen):
    board_part = fen.split(' ')[0]
    rows = board_part.split('/')
    grid = []
    for row in rows:
        grid_row = []
        for char in row:
            if char.isdigit():
                grid_row.extend(['e'] * int(char))
            else:
                grid_row.append(char)
        grid.append(grid_row)
    return grid


def process_dataset(input_root, output_root):
    input_path = Path(input_root)
    output_path = Path(output_root)

    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows_labels = ['8', '7', '6', '5', '4', '3', '2', '1']

    # Constants for 480x480 input
    square_size = 60  # 480 / 8
    tile_size = 96  # Output size
    pad = (tile_size - square_size) // 2  # 18 pixels

    # Support datasets extracted with extra wrapper directories
    # (e.g. content/drive/MyDrive/labeled_data/game*_per_frame).
    for game_dir in input_path.rglob("game*_per_frame"):
        game_id = game_dir.name.replace("_per_frame", "")
        print(f"Processing {game_id}...")

        out_game_dir = output_path / game_id
        out_images_dir = out_game_dir / "tagged_images"
        out_images_dir.mkdir(parents=True, exist_ok=True)

        csv_path = game_dir / f"{game_id}.csv"
        if not csv_path.exists(): continue

        df = pd.read_csv(csv_path)
        gt_data = []

        for _, row in df.iterrows():
            frame_num = str(row['from_frame']).zfill(6)
            img_path = game_dir / "tagged_images" / f"frame_{frame_num}.jpg"

            if not img_path.exists(): continue
            img = cv2.imread(str(img_path))
            if img is None: continue

            # Handle edge padding via reflection
            img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
            grid = fen_to_grid(row['fen'])

            for r in range(8):
                for c in range(8):
                    # Coordinates for slicing from the padded image
                    y_start = r * square_size
                    y_end = y_start + tile_size
                    x_start = c * square_size
                    x_end = x_start + tile_size

                    tile = img_padded[y_start:y_end, x_start:x_end]

                    coord = f"{columns[c]}{rows_labels[r]}"
                    tile_filename = f"{game_id}_{frame_num}_{coord}.jpg"

                    cv2.imwrite(str(out_images_dir / tile_filename), tile)
                    gt_data.append({"file_name": tile_filename, "fen": grid[r][c]})

        pd.DataFrame(gt_data).to_csv(out_game_dir / "gt.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess labeled frames into board-square tiles."
    )
    parser.add_argument(
        "--input-dir",
        default="assets/labeled",
        help="Path to raw labeled dataset (default: assets/labeled).",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/dataset",
        help="Path to write preprocessed tiles (default: assets/dataset).",
    )
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output_dir)