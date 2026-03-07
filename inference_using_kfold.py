import os
import shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# --- GLOBAL MAPPINGS ---
CHAR_TO_UNIFIED = {'e': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                   'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}


# --- MINIMAL DATASET FOR INFERENCE ---
class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Traverse all game directories in the dataset path
        games = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for game in games:
            images_dir = os.path.join(data_dir, game, 'tagged_images')
            if not os.path.exists(images_dir): continue
            for img_name in os.listdir(images_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(os.path.join(images_dir, img_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img), img_path
        except:
            return None


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    imgs, paths = zip(*batch)
    return torch.stack(imgs), paths


# --- MODEL ARCHITECTURE ---
class ConfigurableChessResNet(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        resnet = models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        if self.num_heads == 1:
            self.head_main = nn.Linear(num_ftrs, 13)
        elif self.num_heads == 2:
            self.head_occ = nn.Linear(num_ftrs, 2)
            self.head_piece = nn.Linear(num_ftrs, 12)
        elif self.num_heads == 3:
            self.head_occ = nn.Linear(num_ftrs, 2)
            self.head_color = nn.Linear(num_ftrs, 2)
            self.head_piece = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        if self.num_heads == 1:
            return self.head_main(features)
        elif self.num_heads == 2:
            return self.head_occ(features), self.head_piece(features)
        elif self.num_heads == 3:
            return self.head_occ(features), self.head_color(features), self.head_piece(features)


# --- INFERENCE ENGINE ---
def run_ood_inference(models_path, heads, dataset_path, threshold, device):
    # Setup Output
    output_csv = "low_confidence_images.csv"
    ood_dir = "ood_inspection"
    if os.path.exists(ood_dir): shutil.rmtree(ood_dir)
    os.makedirs(ood_dir)

    # Load Ensemble
    models_list = []
    print(f"Loading models from {models_path}...")
    for file in os.listdir(models_path):
        if file.endswith(".pth"):
            model = ConfigurableChessResNet(heads).to(device)
            model.load_state_dict(torch.load(os.path.join(models_path, file), map_location=device))
            model.eval()
            models_list.append(model)

    if not models_list:
        print("Error: No .pth models found in the specified path.")
        return

    # Prepare Data
    dataset = InferenceDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

    ood_metadata = []
    print(f"Starting inference on {len(dataset)} images...")

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            inputs, paths = batch
            inputs = inputs.to(device)

            ensemble_probs = torch.zeros((inputs.size(0), 13), device=device)

            for model in models_list:
                if heads == 1:
                    ensemble_probs += F.softmax(model(inputs), dim=1)
                elif heads == 2:
                    out_occ, out_piece = model(inputs)
                    p_occ, p_piece = F.softmax(out_occ, dim=1), F.softmax(out_piece, dim=1)
                    unified = torch.zeros((inputs.size(0), 13), device=device)
                    unified[:, 0] = p_occ[:, 0]
                    for i in range(12): unified[:, i + 1] = p_occ[:, 1] * p_piece[:, i]
                    ensemble_probs += unified
                elif heads == 3:
                    out_occ, out_color, out_piece = model(inputs)
                    p_occ, p_color, p_piece = F.softmax(out_occ, dim=1), F.softmax(out_color, dim=1), F.softmax(
                        out_piece, dim=1)
                    unified = torch.zeros((inputs.size(0), 13), device=device)
                    unified[:, 0] = p_occ[:, 0]
                    for i in range(6):
                        unified[:, i + 1] = p_occ[:, 1] * p_color[:, 1] * p_piece[:, i]
                        unified[:, i + 7] = p_occ[:, 1] * p_color[:, 0] * p_piece[:, i]
                    ensemble_probs += unified

            ensemble_probs /= len(models_list)
            max_probs, preds = torch.max(ensemble_probs, dim=1)

            # Identify OOD Samples
            for i in range(inputs.size(0)):
                prob = max_probs[i].item()
                if prob < threshold:
                    img_path = paths[i]
                    filename = os.path.basename(img_path)

                    ood_metadata.append({
                        'file_path': img_path,
                        'confidence': prob,
                        'predicted_class': preds[i].item()
                    })

                    # Copy image for inspection
                    shutil.copy(img_path, os.path.join(ood_dir, f"{prob:.4f}_{filename}"))

    # Save CSV
    df = pd.DataFrame(ood_metadata)
    df.to_csv(output_csv, index=False)
    print(f"\nInference Complete.")
    print(f"OOD Images identified: {len(ood_metadata)}")
    print(f"Metadata saved to: {output_csv}")
    print(f"Inspect images in: {ood_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-Fold Ensemble OOD Inference")
    parser.add_argument("--models_path", type=str, required=True, help="Path to dir containing .pth files")
    parser.add_argument("--heads", type=int, required=True, choices=[1, 2, 3], help="Model config (num_heads)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to assets/new_dataset")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for OOD")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ood_inference(args.models_path, args.heads, args.dataset_path, args.threshold, device)