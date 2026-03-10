#!/usr/bin/env python3
"""Run single-tile inference from an image path using train.py output model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("train_results/BEST_MODEL_train.pth")

IDX_TO_LABEL = {
    0: "empty",
    1: "white_pawn",
    2: "white_knight",
    3: "white_bishop",
    4: "white_rook",
    5: "white_queen",
    6: "white_king",
    7: "black_pawn",
    8: "black_knight",
    9: "black_bishop",
    10: "black_rook",
    11: "black_queen",
    12: "black_king",
}


class ConfigurableChessResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.head_occ = nn.Linear(num_ftrs, 2)
        self.head_piece = nn.Linear(num_ftrs, 12)

    def forward(self, x: torch.Tensor):
        features = torch.flatten(self.backbone(x), 1)
        return self.head_occ(features), self.head_piece(features), features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer a chess tile from an image.")
    parser.add_argument("image_path", type=Path, help="Path to tile image file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Confidence threshold. Below this prints 'ood'.",
    )
    return parser.parse_args()


def preprocess_image(image_path: Path, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def load_model(device: torch.device) -> nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train.py first to create it."
        )
    model = ConfigurableChessResNet().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer(model: nn.Module, tile_tensor: torch.Tensor) -> tuple[int, float]:
    with torch.no_grad():
        out_occ, out_piece, _ = model(tile_tensor)
        p_occ = F.softmax(out_occ, dim=1)
        p_piece = F.softmax(out_piece, dim=1)
        unified = torch.zeros((1, 13), device=tile_tensor.device)
        unified[:, 0] = p_occ[:, 0]
        for i in range(12):
            unified[:, i + 1] = p_occ[:, 1] * p_piece[:, i]
        confidence, idx = torch.max(unified, dim=1)
    return int(idx.item()), float(confidence.item())


def main() -> None:
    args = parse_args()
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_tensor = preprocess_image(args.image_path, device)
    model = load_model(device)
    pred_idx, confidence = infer(model, tile_tensor)

    if confidence < args.threshold:
        print("prediction: ood")
    else:
        print(f"prediction: {IDX_TO_LABEL[pred_idx]}")
    print(f"confidence: {confidence:.4f}")
    print(f"model_used: {MODEL_PATH}")


if __name__ == "__main__":
    main()
