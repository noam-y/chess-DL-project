This report outlines the development of a deep learning system for automated chess piece recognition from video frames.

## 1. Abstract

The problem addressed is the accurate identification of chess board states from high-variance, real-world images where lighting, angles, and piece designs differ across games. The approach utilizes a ResNet-18 backbone modified with a multi-task hierarchical head to predict occupancy, color, and piece type independently. Key results include achieving a Final LOGO (Leave-One-Game-Out) Cross-Validation average accuracy of **94.74% ± 3.89%** across five distinct game environments.

---

## 2. Introduction

* **Task Description and Motivation**: The project aims to translate a physical chessboard image into a digital FEN (Forsyth-Edwards Notation) string. This is motivated by the need for automated game recording and analysis.
* **Challenges and Goals**: The main challenge is domain generalization; models often fail when moved to a new environment with different lighting or board colors. The goal is to create a robust model that generalizes across unseen games.
* **Main Contributions**: Implementation of a multi-head architecture for hierarchical classification and the application of Leave-One-Game-Out (LOGO) cross-validation to ensure true generalization.

---

## 3. Related Work

* **Datasets**: The project utilizes five labeled games of chess data, each with distinct visual characteristics.
* **Prior Work**: This work builds upon standard ResNet-18 classification by splitting the single 13-class output into a multi-head structure to handle the hierarchical nature of chess pieces (Empty vs. Occupied, White vs. Black, and Type).

---

## 4. Method

### Model Architecture

The model uses a **ResNet-18 backbone** (pretrained on ImageNet) truncated before the final fully connected layer. The 512-dimensional feature vector is fed into three separate linear heads:

* **Occupancy Head**: 2 classes (Empty, Occupied)
* **Color Head**: 2 classes (Black, White)
* **Piece Head**: 6 classes (Pawn, Knight, Bishop, Rook, Queen, King)

### Input/Output Representation

* **Input**: 96x96 pixel square patches cropped from the board.
* **Output**: A combined FEN string representing the 8x8 board state.

### Training Procedure & Loss Functions

* **Training**: Adam optimizer ($lr=0.001$) with a `ReduceLROnPlateau` scheduler.
* **Loss Function**: A **Masked Hierarchical Cross-Entropy Loss**.

$$L_{total} = L_{occ} + \mathbb{1}_{[occupied]} (L_{color} + L_{piece})$$


* **Class Weighting**: To handle imbalance (many empty squares, few Queens), inverse frequency weights were applied to the loss functions:
* **Queen Weight**: 3.133
* **Pawn Weight**: 0.319



---

## 5. Experiments

### Dataset Description and Splits

The data consists of 5 labeled games (e.g., `game2`, `game4`).

* **Split**: Leave-One-Game-Out (LOGO) Cross-Validation. In each fold, the model is trained on 4 games and validated on the 5th entirely unseen game.

### Evaluation Metrics

* **Piece Accuracy**: Percentage of individual squares correctly identified.
* **Board Accuracy**: Percentage of full boards (64 squares) identified perfectly.

### Quantitative Results

| Fold (Validation Game) | Hierarchical Accuracy |
| --- | --- |
| Game 2 | 96.92% |
| Game 4 | 95.75% |
| Game 5 | 98.34% |
| Game 6 | 87.23% |
| Game 7 | 95.45% |
| **Mean Accuracy** | **94.74% ± 3.89%** |

---

## 6. Ablation Study

* **Multi-Head vs. Single Head**: Splitting the task into Occupancy/Color/Piece allows the model to learn "Empty" features separately from "Piece" features, preventing the model from confusing empty squares with specific piece types.
* **Loss Masking**: Removing the mask (calculating color loss on empty squares) forced the model to learn arbitrary colors for empty wood, degrading performance on actual pieces.
* **Class Weighting**: Removing class weights caused the model to frequently misidentify rare pieces (Queens/Kings) as common ones (Pawns) or empty squares.

---

## 7. What Did Not Work

* **Triplet Loss**: Initially considered to learn embeddings, but was abandoned due to the high computational overhead in the data loader and the efficiency of the hierarchical head approach.
* **Softmax Temperature**: Tried during training to regularize confidence but provided no benefit over standard Cross-Entropy with class weights.
* **Isolated Patching**: Without the 1.5x crop padding, the model struggled with tall pieces (Kings/Queens) that occluded adjacent squares.

---

## 8. Discussion / Limitations

* **Failure Cases**: The model struggles with **Hand Occlusion**, where a player moving a piece blocks the board.
* **Perspective Distortion**: The 8x8 linear grid assumption fails when the camera is at a sharp angle, causing square misalignment.
* **Future Improvements**: Implementing a Homography (Perspective Transform) to flatten the board before patching and adding a hand-detection mask to suppress inference on occluded squares.

Would you like me to help you generate the specific LaTeX equations for the Method section?