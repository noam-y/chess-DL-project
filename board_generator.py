import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# --- Configuration ---
INPUT_DIR = 'BASE_TO_AUGMENT'
OUTPUT_DIR = 'new_augmented_data'
IMG_SIZE = 480
PATCH_SIZE = 60  

def augment_chess_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in {INPUT_DIR}")
        return

    tensors = []
    for f in files:
        img_path = os.path.join(INPUT_DIR, f)
        img = Image.open(img_path).convert('RGB')
        tensors.append(transform(img))
    
    batch = torch.stack(tensors) 
    B, C, H, W = batch.shape
    
    print(f"Loaded batch shape: {batch.shape}")

    # Unfold
    patches = F.unfold(batch, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    num_patches = patches.shape[2]

    mixed_patches_list = []
    
    for i in range(B):
        source_indices = torch.randint(0, B, (num_patches,))
        
        # --- התיקון נמצא בשורה הבאה ---
        # השליפה מחזירה (64, 10800), אנחנו צריכים להפוך ל-(10800, 64)
        # אז הוספנו .t() (transpose) בסוף
        mixed_img_patches = patches[source_indices, :, torch.arange(num_patches)].t()
        
        mixed_patches_list.append(mixed_img_patches)

    mixed_batch_unfolded = torch.stack(mixed_patches_list)

    # Fold
    new_images = F.fold(
        mixed_batch_unfolded, 
        output_size=(H, W), 
        kernel_size=PATCH_SIZE, 
        stride=PATCH_SIZE
    )

    print(f"Saving {B} augmented images to '{OUTPUT_DIR}'...")
    for i in range(B):
        output_path = os.path.join(OUTPUT_DIR, f"aug_{i}_{files[i]}")
        save_image(new_images[i], output_path)

    print("Done.")

if __name__ == "__main__":
    augment_chess_data()