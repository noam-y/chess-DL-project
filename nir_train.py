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
        
        label = PIECE_TO_ID.get(label_str, 0) # Default to 0 (empty)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 3. Model Definition (Embedder + Classifier) ---
class ResNetWithEmbeddings(nn.Module):
    def __init__(self, num_classes=13):
        super(ResNetWithEmbeddings, self).__init__()
        # Load pretrained ResNet18
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the final FC layer to get the feature extractor
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Get input dimension for the FC layer (usually 512 for ResNet18)
        num_ftrs = base_model.fc.in_features
        
        # New classification head
        self.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # Flatten (batch_size, 512)
        
        embeddings = x  # These are the features we'll use for Triplet Loss
        logits = self.fc(x) # These are for CrossEntropy / Classification
        
        return logits, embeddings

# --- 4. Triplet Loss Implementation (Batch Hard) ---
def batch_hard_triplet_loss(embeddings, labels, margin=1.0, device='cpu'):
    """
    Computes Batch Hard Triplet Loss.
    For each anchor, pick the hardest positive (furthest) and hardest negative (closest).
    """
    # Normalize embeddings (optional but recommended for triplet loss)
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
        # Hardest Positive: Max distance among positives
        pos_dists = dists[i][pos_mask[i]]
        if len(pos_dists) == 0:
            continue # No positive pair for this anchor in this batch
        hardest_pos = torch.max(pos_dists)
        
        # Hardest Negative: Min distance among negatives
        neg_dists = dists[i][neg_mask[i]]
        if len(neg_dists) == 0:
            continue # Should not happen if batch has >1 class
        hardest_neg = torch.min(neg_dists)
        
        # Loss = max(0, d_pos - d_neg + margin)
        current_loss = torch.relu(hardest_pos - hardest_neg + margin)
        loss += current_loss
        valid_triplets += 1
        
    if valid_triplets > 0:
        loss /= valid_triplets
        
    return loss

# --- 5. Centroid Computation & OOD Logic ---
def compute_centroids(model, dataloader, device, num_classes=13):
    print("Computing class centroids (from Train Set)...")
    model.eval()
    
    centroids = torch.zeros(num_classes, 512).to(device)
    class_counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Computing Centroids"):
            images = images.to(device)
            labels = labels.to(device)
            
            _, embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            for i in range(len(labels)):
                label = labels[i]
                centroids[label] += embeddings[i]
                class_counts[label] += 1
                
    # Normalize
    for c in range(num_classes):
        if class_counts[c] > 0:
            centroids[c] /= class_counts[c]
            # Re-normalize centroid to unit sphere
            centroids[c] = F.normalize(centroids[c].unsqueeze(0), p=2, dim=1).squeeze(0)
            
    return centroids

def evaluate_model(model, dataloader, device, set_name="Validation"):
    print(f"\nRunning Evaluation on {set_name} Set...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {set_name}"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits, _ = model(images)
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    print(f"\n{set_name} Set Detailed Report:")
    
    # Use unique labels present in the targets to avoid index errors if some classes are missing
    unique_labels = sorted(list(set(all_targets)))
    target_names_present = [ID_TO_PIECE[i] for i in unique_labels]
    
    print(classification_report(all_targets, all_preds, zero_division=0, target_names=target_names_present, labels=unique_labels))
    
    print(f"{set_name} Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds, labels=unique_labels))
    
    # Return overall accuracy for simple tracking
    correct = sum(p == t for p, t in zip(all_preds, all_targets))
    return 100.0 * correct / len(all_targets)

# --- 6. Main Training Function ---
def main(args):
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Starting training on: {device}")
    
    # Directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset Loading & Splitting
    full_dataset = ChessTilesDataset(root_dir=args.data_dir, transform=train_transform)
    if len(full_dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # 80-20 Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Data Split: {train_size} Training samples, {val_size} Validation samples.")

    # DataLoaders
    # Note: drop_last=True for train to help triplet mining. 
    # For validation, we want to evaluate everything, so drop_last=False.
    batch_size = max(args.batch_size, 8) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        drop_last=True 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # Model
    print("Initializing ResNet18 with Triplet Loss support...")
    model = ResNetWithEmbeddings(num_classes=13).to(device)

    # Losses
    class_weights = torch.ones(13)
    class_weights[0] = 0.1 
    class_weights = class_weights.to(device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    triplet_margin = 1.0
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_ce_loss = 0.0
        running_triplet_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits, embeddings = model(images)
            
            # Compute Losses
            ce_loss = ce_criterion(logits, labels)
            tri_loss = batch_hard_triplet_loss(embeddings, labels, margin=triplet_margin, device=device)
            
            # Combined Loss
            total_loss = ce_loss + tri_loss
            
            total_loss.backward()
            optimizer.step()

            # Stats
            running_ce_loss += ce_loss.item()
            running_triplet_loss += tri_loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(
                ce_loss=f"{ce_loss.item():.3f}", 
                tri_loss=f"{tri_loss.item():.3f}", 
                acc=f"{100.*correct/total:.1f}%"
            )

        # End of epoch stats
        epoch_ce = running_ce_loss / len(train_loader)
        epoch_tri = running_triplet_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1} Training Summary: CE_Loss={epoch_ce:.4f}, Triplet_Loss={epoch_tri:.4f}, Train_Acc={epoch_acc:.2f}%")
        
        # --- Validation Step ---
        val_acc = evaluate_model(model, val_loader, device, set_name="Validation")
        print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}%")
        
        # Save model
        save_path = os.path.join(args.output_dir, f"resnet18_triplet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Compute and Save Centroids (Using Training Set)
    centroids = compute_centroids(model, train_loader, device)
    torch.save(centroids, os.path.join(args.output_dir, "centroids.pt"))
    print(f"Centroids saved to {os.path.join(args.output_dir, 'centroids.pt')}")

    # OOD Demonstration (on Validation set to be fair)
    print(f"\nOOD Statistics on Validation Set (Threshold: {args.ood_threshold}):")
    distances = []
    ood_count = 0
    with torch.no_grad():
         for images, labels in tqdm(val_loader, desc="Checking OOD stats"):
            images = images.to(device)
            _, embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Batch distance calculation
            dists = torch.cdist(embeddings, centroids.unsqueeze(0), p=2).squeeze(0)
            min_dists, _ = torch.min(dists, dim=1)
            
            distances.extend(min_dists.cpu().numpy())
            ood_count += (min_dists > args.ood_threshold).sum().item()
            
    avg_dist = np.mean(distances)
    max_dist = np.max(distances)
    p95_dist = np.percentile(distances, 95)
    p99_dist = np.percentile(distances, 99)
    
    print(f"Average Distance to Nearest Centroid: {avg_dist:.4f}")
    print(f"95th Percentile Distance: {p95_dist:.4f}")
    print(f"99th Percentile Distance: {p99_dist:.4f}")
    print(f"Max Distance to Nearest Centroid: {max_dist:.4f}")
    print(f"Flagged as OOD (Threshold > {args.ood_threshold}): {ood_count} / {len(val_dataset)} ({100.0 * ood_count / len(val_dataset):.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet with Triplet Loss on Chess Tiles")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_resnet_triplet", help="Where to save the model")
    
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--ood_threshold", type=float, default=0.9, help="Distance threshold for OOD detection")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'auto', 'cuda', 'cpu'")

    args = parser.parse_args()
    
    main(args)
