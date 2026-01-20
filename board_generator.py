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
PATCH_SIZE = 60  # 480 / 8 = 60 pixels per square

def augment_chess_data():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Images to Tensor
    # Transform: Resize -> Convert to Tensor (scales to [0, 1])
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No images found in {INPUT_DIR}")
        return

    # Create a batch tensor: (Batch_Size, Channels, Height, Width)
    tensors = []
    for f in files:
        img_path = os.path.join(INPUT_DIR, f)
        img = Image.open(img_path).convert('RGB')
        tensors.append(transform(img))
    
    batch = torch.stack(tensors)  # Shape: (B, 3, 480, 480)
    B, C, H, W = batch.shape
    
    print(f"Loaded batch shape: {batch.shape}")

    # 2. Patch Extraction (Unfold)
    # Extracts patches. Shape becomes: (B, C*Patch_H*Patch_W, Num_Patches)
    patches = F.unfold(batch, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    num_patches = patches.shape[2]  # 64 patches for 8x8 grid

    # 3. Shuffle Patches Logic
    mixed_patches_list = []
    
    for i in range(B):
        # For each spatial position (0 to 63), pick a random image index (0 to B-1)
        source_indices = torch.randint(0, B, (num_patches,))
        
        # Advanced Indexing: Gather specific patches from different images
        # patches shape: (Batch, Vectorized_Patch, Patch_ID)
        # We select: (source_indices, :, 0...63)
        mixed_img_patches = patches[source_indices, :, torch.arange(num_patches)]
        mixed_patches_list.append(mixed_img_patches)

    # Stack back to batch: (B, Vectorized_Patch, Num_Patches)
    mixed_batch_unfolded = torch.stack(mixed_patches_list)

    # 4. Reconstruct Images (Fold)
    # Inverse operation to create full images from patches
    new_images = F.fold(
        mixed_batch_unfolded, 
        output_size=(H, W), 
        kernel_size=PATCH_SIZE, 
        stride=PATCH_SIZE
    )

    # 5. Save to Disk
    print(f"Saving {B} augmented images to '{OUTPUT_DIR}'...")
    for i in range(B):
        output_path = os.path.join(OUTPUT_DIR, f"aug_{i}_{files[i]}")
        # save_image handles un-normalizing if needed (assumes [0,1] input)
        save_image(new_images[i], output_path)

    print("Done.")

if __name__ == "__main__":
    augment_chess_data()