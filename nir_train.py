import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- 1. Utility & Configuration ---
PIECE_TO_ID = {
    'e': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
}

ID_TO_PIECE = {v: k for k, v in PIECE_TO_ID.items()}

# Color Mapping: 0=Empty, 1=White, 2=Black
def get_color_label(piece_char):
    if piece_char == 'e':
        return 0
    elif piece_char.isupper():
        return 1 # White
    else:
        return 2 # Black

# --- 2. Dataset Definition ---
class ChessTilesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'images')
        self.gt_path = os.path.join(root_dir, 'gt.csv')
        
        if not os.path.exists(self.gt_path):
            raise FileNotFoundError(f"Ground truth file not found at {self.gt_path}")
            
        self.df = pd.read_csv(self.gt_path)
        print(f"Loaded dataset from {root_dir} with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_name']
        label_str = row['piece_id']
        
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Piece Label (0-12)
        piece_label = PIECE_TO_ID.get(label_str, 0)
        
        # Color Label (0-2)
        color_label = get_color_label(label_str)
        
        if self.transform:
            image = self.transform(image)
            
        return image, piece_label, color_label

# --- 3. Model Definition (Embedder + 2 Classifiers) ---
class ResNetMultiHead(nn.Module):
    def __init__(self, num_piece_classes=13, num_color_classes=3):
        super(ResNetMultiHead, self).__init__()
        # Load pretrained ResNet18
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the final FC layer to get the feature extractor
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Get input dimension for the FC layer (usually 512 for ResNet18)
        num_ftrs = base_model.fc.in_features
        
        # Head 1: Piece Classification (13 classes)
        self.fc_piece = nn.Linear(num_ftrs, num_piece_classes)
        
        # Head 2: Color Classification (3 classes)
        self.fc_color = nn.Linear(num_ftrs, num_color_classes)
        
    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # Flatten (batch_size, 512)
        
        embeddings = x  # Features for Triplet Loss
        
        logits_piece = self.fc_piece(x)
        logits_color = self.fc_color(x)
        
        return logits_piece, logits_color, embeddings

# --- 4. Triplet Loss Implementation (Batch Hard) ---
def batch_hard_triplet_loss(embeddings, labels, margin=1.0, device='cpu'):
    """
    Computes Batch Hard Triplet Loss based on PIECE labels.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Pairwise Euclidean distances
    dists = torch.cdist(embeddings, embeddings, p=2)
    
    # Masks
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1) # (B, B)
    eye = torch.eye(len(labels), dtype=torch.bool, device=device)
    
    pos_mask = labels_eq & (~eye) # Same label, different index
    neg_mask = ~labels_eq         # Different label
    
    batch_size = len(labels)
    loss = torch.tensor(0.0, device=device)
    valid_triplets = 0
    
    for i in range(batch_size):
        pos_dists = dists[i][pos_mask[i]]
        if len(pos_dists) == 0: continue
        hardest_pos = torch.max(pos_dists)
        
        neg_dists = dists[i][neg_mask[i]]
        if len(neg_dists) == 0: continue
        hardest_neg = torch.min(neg_dists)
        
        current_loss = torch.relu(hardest_pos - hardest_neg + margin)
        loss += current_loss
        valid_triplets += 1
        
    if valid_triplets > 0:
        loss /= valid_triplets
        
    return loss

# --- 5. Centroid Computation & Evaluation ---
def compute_centroids(model, dataloader, device, num_classes=13):
    print("Computing class centroids (from Train Set)...")
    model.eval()
    
    centroids = torch.zeros(num_classes, 512).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for images, piece_labels, _ in tqdm(dataloader, desc="Computing Centroids"):
            images = images.to(device)
            piece_labels = piece_labels.to(device)
            
            _, _, embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            for i in range(len(piece_labels)):
                label = piece_labels[i]
                centroids[label] += embeddings[i]
                class_counts[label] += 1
                
    for c in range(num_classes):
        if class_counts[c] > 0:
            centroids[c] /= class_counts[c]
            centroids[c] = F.normalize(centroids[c].unsqueeze(0), p=2, dim=1).squeeze(0)
            
    return centroids

def evaluate_model(model, dataloader, device, set_name="Validation"):
    print(f"\nRunning Evaluation on {set_name} Set...")
    model.eval()
    
    all_preds_piece = []
    all_targets_piece = []
    all_preds_color = []
    all_targets_color = []
    
    with torch.no_grad():
        for images, piece_labels, color_labels in tqdm(dataloader, desc=f"Evaluating {set_name}"):
            images = images.to(device)
            piece_labels = piece_labels.to(device)
            color_labels = color_labels.to(device)
            
            logits_piece, logits_color, _ = model(images)
            
            _, pred_piece = torch.max(logits_piece.data, 1)
            _, pred_color = torch.max(logits_color.data, 1)
            
            all_preds_piece.extend(pred_piece.cpu().numpy())
            all_targets_piece.extend(piece_labels.cpu().numpy())
            
            all_preds_color.extend(pred_color.cpu().numpy())
            all_targets_color.extend(color_labels.cpu().numpy())

    print(f"\n--- {set_name} PIECE Classification Report ---")
    unique_labels = sorted(list(set(all_targets_piece)))
    target_names_present = [ID_TO_PIECE[i] for i in unique_labels]
    print(classification_report(all_targets_piece, all_preds_piece, zero_division=0, target_names=target_names_present, labels=unique_labels))
    
    print(f"\n--- {set_name} COLOR Classification Report ---")
    print(classification_report(all_targets_color, all_preds_color, zero_division=0, target_names=['Empty', 'White', 'Black']))
    
    # Return piece accuracy for tracking best model
    correct = sum(p == t for p, t in zip(all_preds_piece, all_targets_piece))
    return 100.0 * correct / len(all_targets_piece)

# --- 6. Main Training Function ---
def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Starting training on: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset Loading
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Train/Test directories not found in {args.data_dir}")
        return

    train_dataset = ChessTilesDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = ChessTilesDataset(root_dir=test_dir, transform=val_transform)
    
    print(f"Data Loaded: {len(train_dataset)} Training samples, {len(val_dataset)} Validation/Test samples.")

    # Assign transforms to subsets (Wrapper logic)
    class SubsetWithTransform(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            # Underlying dataset returns (img, piece, color)
            img, piece, color = self.subset[idx]
            if self.transform: img = self.transform(img)
            return img, piece, color

    # Since we loaded separate datasets with transforms already, we don't need the wrapper here
    # unless we want to split the *train_dataset* further for validation.
    # But the prompt implies using the 'test' folder as validation.
    # Let's stick to train_dataset for training and val_dataset for validation.
    
    batch_size = max(args.batch_size, 8) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Model
    print("Initializing ResNet18 MultiHead...")
    model = ResNetMultiHead(num_piece_classes=13, num_color_classes=3).to(device)

    # Losses
    # Piece Loss (Weighted)
    piece_weights = torch.ones(13).to(device)
    piece_weights[0] = 0.1 
    criterion_piece = nn.CrossEntropyLoss(weight=piece_weights)
    
    # Color Loss (Weighted - Empty is dominant)
    color_weights = torch.tensor([0.1, 1.0, 1.0]).to(device)
    criterion_color = nn.CrossEntropyLoss(weight=color_weights)
    
    triplet_margin = 1.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_val_acc = 0.0

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss_p = 0.0
        running_loss_c = 0.0
        running_loss_t = 0.0
        correct_p = 0
        correct_c = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, piece_labels, color_labels in loop:
            images = images.to(device)
            piece_labels = piece_labels.to(device)
            color_labels = color_labels.to(device)
            
            optimizer.zero_grad()
            
            logits_piece, logits_color, embeddings = model(images)
            
            loss_piece = criterion_piece(logits_piece, piece_labels)
            loss_color = criterion_color(logits_color, color_labels)
            loss_triplet = batch_hard_triplet_loss(embeddings, piece_labels, margin=triplet_margin, device=device)
            
            total_loss = loss_piece + loss_color + loss_triplet
            total_loss.backward()
            optimizer.step()

            # Stats
            running_loss_p += loss_piece.item()
            running_loss_c += loss_color.item()
            running_loss_t += loss_triplet.item()
            
            _, pred_p = torch.max(logits_piece.data, 1)
            _, pred_c = torch.max(logits_color.data, 1)
            
            total += piece_labels.size(0)
            correct_p += (pred_p == piece_labels).sum().item()
            correct_c += (pred_c == color_labels).sum().item()
            
            loop.set_postfix(
                lp=f"{loss_piece.item():.2f}", 
                lc=f"{loss_color.item():.2f}", 
                lt=f"{loss_triplet.item():.2f}",
                acc_p=f"{100.*correct_p/total:.1f}%"
            )

        # Validation
        val_acc = evaluate_model(model, val_loader, device, set_name="Validation")
        print(f"Epoch {epoch+1} Validation Piece Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_filename = f"resnet18_best_bs{args.batch_size}_epoch{epoch+1}_acc{val_acc:.2f}.pth"
            save_path = os.path.join(args.output_dir, model_filename)
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
            
            latest_path = os.path.join(args.output_dir, "resnet18_best.pth")
            torch.save(model.state_dict(), latest_path)

    # OOD Stats (using best model)
    best_model_path = os.path.join(args.output_dir, "resnet18_best.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    centroids = compute_centroids(model, train_loader, device)
    torch.save(centroids, os.path.join(args.output_dir, "centroids.pt"))
    
    # OOD Check
    print(f"\nOOD Statistics (Threshold: {args.ood_threshold}):")
    distances = []
    ood_count = 0
    with torch.no_grad():
         for images, _, _ in tqdm(val_loader, desc="Checking OOD stats"):
            images = images.to(device)
            _, _, embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            dists = torch.cdist(embeddings, centroids.unsqueeze(0), p=2).squeeze(0)
            min_dists, _ = torch.min(dists, dim=1)
            distances.extend(min_dists.cpu().numpy())
            ood_count += (min_dists > args.ood_threshold).sum().item()
            
    print(f"Avg Dist: {np.mean(distances):.4f}, Max Dist: {np.max(distances):.4f}")
    print(f"99th Percentile: {np.percentile(distances, 99):.4f}")
    print(f"Flagged OOD: {ood_count}/{len(val_dataset)} ({100.0*ood_count/len(val_dataset):.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet MultiHead")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_resnet_multihead")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--ood_threshold", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    main(args)
