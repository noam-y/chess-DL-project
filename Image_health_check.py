import os
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def main():
    data_dir = Path("assets/new_dataset")

    # Trackers
    total_images_in_csv = 0
    missing_files = 0
    corrupted_files = 0

    print("Starting Dataset Health Inspection...\n")

    # Find all games
    game_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("game")]

    for game_dir in sorted(game_dirs):
        csv_path = game_dir / "gt.csv"
        if not csv_path.exists():
            print(f"[{game_dir.name}] No gt.csv found! Skipping.")
            continue

        df = pd.read_csv(csv_path)
        game_total = len(df)
        total_images_in_csv += game_total

        game_missing = 0
        game_corrupted = 0

        images_dir = game_dir / "tagged_images"

        # Wrap the iteration in a tqdm progress bar for this game
        for _, row in tqdm(df.iterrows(), total=game_total, desc=f"Scanning {game_dir.name}"):
            img_path = images_dir / row['file_name']

            # 1. Check if file physically exists
            if not img_path.exists():
                game_missing += 1
                missing_files += 1
                continue

            # 2. Check if file is corrupted (zero-byte or broken JPEG)
            try:
                # .verify() is much faster than loading the whole image array
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                game_corrupted += 1
                corrupted_files += 1

        if game_missing > 0 or game_corrupted > 0:
            print(
                f"  -> WARNING in {game_dir.name}: {game_missing} missing, {game_corrupted} corrupted out of {game_total} tiles.")
        else:
            print(f"  -> {game_dir.name} is 100% healthy! ({game_total} tiles)")

    # --- FINAL REPORT ---
    print("\n" + "=" * 50)
    print("FINAL DATASET HEALTH REPORT")
    print("=" * 50)
    print(f"Total Images Expected (from CSVs): {total_images_in_csv}")
    print(f"Missing Files:                     {missing_files}")
    print(f"Corrupted Files:                   {corrupted_files}")

    healthy_files = total_images_in_csv - missing_files - corrupted_files
    health_percentage = (healthy_files / total_images_in_csv) * 100 if total_images_in_csv > 0 else 0

    print(f"\nOverall Dataset Health:            {health_percentage:.2f}%")
    print("=" * 50)

    if health_percentage < 99.0:
        print(
            "\nACTION REQUIRED: Your dataset has significant damage. You may need to re-run the offline generation script.")
    elif health_percentage < 100.0:
        print(
            "\nMINOR ISSUE: A tiny fraction of images are bad. The custom_collate function will safely ignore them. You are good to train!")
    else:
        print("\nPERFECT: Your dataset is flawless. You are good to train!")


if __name__ == "__main__":
    main()