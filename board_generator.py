import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

INPUT_DIR = 'BASE_TO_AUGMENT'
OUTPUT_DIR = 'new_augmented_data'
IMG_SIZE = 480
PATCH_SIZE = 60 

def augment_data():
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
    
    # Shape: (Batch_Size, 3, 480, 480)
    batch = torch.stack(tensors)
    B, C, H, W = batch.shape
    
    print(f"Processing batch: {batch.shape}")

    # 2. Extract Patches (Unfold)
    # Result: (B, Channels*PatchH*PatchW, Num_Patches)
    patches = F.unfold(batch, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    num_patches = patches.shape[2] 

    # 3. Shuffle Patches (Inter-image mixing)
    mixed_list = []
    for i in range(B):
        source_idx = torch.randint(0, B, (num_patches,))
        
        # Advanced indexing: Result is (Num_Patches, Flattened_Patch_Size)
        # We use .t() to transpose back to (Flattened_Patch_Size, Num_Patches) for Fold
        mixed = patches[source_idx, :, torch.arange(num_patches)].t()
        mixed_list.append(mixed)

    mixed_batch = torch.stack(mixed_list)

    new_images = F.fold(
        mixed_batch, 
        output_size=(H, W), 
        kernel_size=PATCH_SIZE, 
        stride=PATCH_SIZE
    )

    print(f"Saving {B} images to {OUTPUT_DIR}...")
    for i in range(B):
        save_image(new_images[i], os.path.join(OUTPUT_DIR, f"aug_{i}.png"))
    
    print("Done.")

if __name__ == "__main__":
    augment_data()