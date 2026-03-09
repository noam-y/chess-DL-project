import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

INPUT_DIR = 'aug/BASE_TO_AUGMENT'          
OUTPUT_DIR = 'aug/new_augmented_data'      
LABELS_FILE = 'aug/game6.csv'            
IMG_SIZE = 480
PATCH_SIZE = 60 

def expand_fen_to_list(fen):
    board_part = fen.split(' ')[0]
    rows = board_part.split('/')
    expanded = []
    for row in rows:
        for char in row:
            if char.isdigit():
                expanded.extend(['1'] * int(char)) # '1' ייצג משבצת ריקה
            else:
                expanded.append(char)
    return expanded

def collapse_list_to_fen(char_list):
    fen_rows = []
    for i in range(8):
        row_chars = char_list[i*8 : (i+1)*8]
        new_row = ""
        empty_count = 0
        for char in row_chars:
            if char == '1': 
                empty_count += 1
            else:
                if empty_count > 0:
                    new_row += str(empty_count)
                    empty_count = 0
                new_row += char
        if empty_count > 0:
            new_row += str(empty_count)
        fen_rows.append(new_row)
    return "/".join(fen_rows)

def extract_frame_number(filename):
    """
    מנסה לחלץ את המספר מתוך שם הקובץ.
    למשל: 'frame_36.png' -> 36
    """
    # מוצא את כל רצפי המספרים בשם הקובץ
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # לוקח את האחרון (בדרך כלל זה המספר הסידורי) או הראשון, תלוי בשמות שלך
        # כאן אני מניח שיש מספר אחד משמעותי
        return int(numbers[-1]) 
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(LABELS_FILE):
        print(f"Error: {LABELS_FILE} not found.")
        return
    
    df = pd.read_csv(LABELS_FILE)
    
    frame_to_chars = {}
    for _, row in df.iterrows():
        frame_num = int(row['from_frame'])
        frame_to_chars[frame_num] = expand_fen_to_list(row['fen'])

    print(f"Loaded labels for {len(frame_to_chars)} frames from CSV.")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    img_tensors = []
    label_lists = [] 
    
    print("Matching images to labels...")
    matched_count = 0
    
    for f in files:
        frame_num = extract_frame_number(f)
        
        if frame_num is not None and frame_num in frame_to_chars:
            img_path = os.path.join(INPUT_DIR, f)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensors.append(transform(img))
                label_lists.append(frame_to_chars[frame_num])
                matched_count += 1
            except Exception as e:
                print(f"Error reading {f}: {e}")
        else:
            pass

    if not img_tensors:
        print("No valid image-label pairs found! Check your filenames vs CSV 'from_frame'.")
        return

    print(f"Successfully matched {matched_count} images.")

    # מכאן והלאה - הלוגיקה זהה: המרה לטנזורים, ערבוב ושמירה
    
    # המרת תווים למספרים (Vocab)
    unique_chars = sorted(list(set([c for l in label_lists for c in l])))
    char_to_int = {c: i for i, c in enumerate(unique_chars)}
    int_to_char = {i: c for c, i in char_to_int.items()}
    
    label_tensor = torch.tensor([[char_to_int[c] for c in l] for l in label_lists])
    img_batch = torch.stack(img_tensors) # (B, 3, H, W)

    B, C, H, W = img_batch.shape
    
    # Unfold Images
    patches = F.unfold(img_batch, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    num_patches = patches.shape[2] 

    # Mixing Loop
    mixed_patches_list = []
    new_csv_data = []

    print("Generating synthetic dataset...")
    for i in range(B):
        source_indices = torch.randint(0, B, (num_patches,))
        
        mixed_img = patches[source_indices, :, torch.arange(num_patches)].t()
        mixed_patches_list.append(mixed_img)

        mixed_label_ints = label_tensor[source_indices, torch.arange(num_patches)]
        
        mixed_chars = [int_to_char[idx.item()] for idx in mixed_label_ints]
        new_fen = collapse_list_to_fen(mixed_chars)
        
        filename = f"aug_{i}.png"
        new_csv_data.append({'filename': filename, 'fen': new_fen})

    mixed_batch = torch.stack(mixed_patches_list)
    new_images = F.fold(mixed_batch, output_size=(H, W), kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    print(f"Saving results to {OUTPUT_DIR}...")
    for i in range(B):
        save_image(new_images[i], os.path.join(OUTPUT_DIR, new_csv_data[i]['filename']))

    result_csv_path = os.path.join(OUTPUT_DIR, 'augmented_ground_truth.csv')
    pd.DataFrame(new_csv_data).to_csv(result_csv_path, index=False)
    
    print(f"Done! Created images and {result_csv_path}")

if __name__ == "__main__":
    main()
