This outline summarizes the context of your **Chess Deep Learning project** based on your recent experiments and implementation details. You can use this as a direct draft for your report.

---

# Chess Piece Recognition and OOD Detection

## 1. Abstract

This project addresses the challenge of precise chess piece classification from board-square images. We implemented an ensemble of **ResNet-18** models trained using a **K-Fold cross-validation** strategy. Our approach leverages **Multi-Similarity Loss** and a configurable multi-head architecture to improve classification accuracy and **Out-of-Distribution (OOD)** detection. Our best configuration achieved an **Ensemble Test F1-score of 84.02%** on unseen game data, significantly outperforming standard Batch Hard Triplet Loss baselines.

---

## 2. Introduction

* **Task Description:** Automated recognition of chess pieces ($32$ pieces across $64$ squares) from raw images of a physical chessboard.
* **Motivation:** To create a robust system for digitizing physical games into FEN (Forsyth-Edwards Notation) for analysis or broadcasting.
* **Challenges:** Variable lighting, different piece styles, camera angles, and the high visual similarity between certain pieces (e.g., Pawn vs. Bishop in low-res).
* **Main Contributions:** 1.  Development of a **multi-head ResNet architecture** (Occupancy, Color, Piece Type).
2.  Integration of **Multi-Similarity Loss** for superior metric learning.
3.  A robust **OOD detection pipeline** using ensemble confidence thresholds.

---

## 3. Related Work

* **Architectures:** ResNet-18 was selected as the backbone due to its balance of computational efficiency on cluster nodes and feature extraction depth.
* **Loss Functions:** We compared standard **Batch Hard Triplet Loss** against modern **Multi-Similarity Loss** (inspired by deep metric learning research for fine-grained retrieval).
* **Datasets:** Custom dataset consisting of multiple "Games" (e.g., `game2`, `game4`, `game6`, `game7` for training/validation and `game5` for held-out testing).

---

## 4. Method

* **Model Architecture:** A `ConfigurableChessResNet` using a ResNet-18 backbone.
* **Head 1:** Unified 13-class classifier (Empty + 12 piece types).
* **Head 2:** Split into Occupancy (2) and Piece Type (12).
* **Head 3:** Split into Occupancy (2), Color (2), and Piece Type (6).


* **Loss Functions:**
* **Cross-Entropy Loss** for classification.
* **Metric Learning:** Multi-Similarity Loss applied to the feature embeddings to maximize intra-class similarity and minimize inter-class similarity.


* **Preprocessing:** Images resized to $96 \times 96$, normalized using ImageNet statistics.

---

## 5. Experiments

* **Dataset Splits:** 4-Fold Cross-Validation where each fold leaves out one game (`game2`, `game4`, `game6`, or `game7`) for validation. Final testing performed on `game5`.
* **Evaluation Metrics:** Macro F1-score (to account for class imbalance between "Empty" squares and specific pieces like Kings).
* **Quantitative Results:**
* **Baseline (Old Triplet Loss):** ~72.02% - 75.88% F1.
* **Our Best (New MS-Loss + B16):** **84.02% F1.**



| Configuration | Batch Size | Mean 4-Fold F1 | Ensemble Test F1 |
| --- | --- | --- | --- |
| **H1_MS-Loss_Frozen_B16** | **16** | **56.75%** | **84.02%** |
| H1_MS-Loss_Unfrozen_B32 | 32 | 55.03% | 83.81% |

---

## 6. Ablation Study

* **Loss Function (Triplet vs. Multi-Similarity):** Switching to Multi-Similarity Loss provided a ~10% absolute gain in F1-score.
* **Backbone Freezing:** Freezing the backbone (using pre-trained ImageNet weights) provided more stable convergence and slightly higher test scores compared to full fine-tuning on our smaller dataset.
* **Batch Size:** Reducing batch size from 64 to 16 improved performance by ~1.5%, likely due to more frequent gradient updates and the nature of Multi-Similarity mining.

---

## 7. What Did Not Work

* **Large Batch Sizes (B-64):** Underperformed compared to smaller batches, potentially over-smoothing the gradients in the metric learning stage.
* **Standard Triplet Loss:** Struggled with the "Empty" class vs. "Pawn" class ambiguity, leading to higher OOD rates for brown/dark wood squares.
* **Full Fine-tuning early on:** Led to lower validation scores (Mean CV F1 ~47%) compared to the frozen backbone approach.

---

## 8. Discussion / Limitations

* **Failure Cases:** OOD analysis via `low_confidence_images.csv` showed that the model struggles when a piece is positioned at the very edge of a cropped square.
* **Limitations:** The current model is sensitive to specific board textures not present in the training folds.
* **Future Work:** Implementing **Test-Time Augmentation (TTA)** and using **Visual Transformers (ViT)** for better global context of the board.

---

### Next Steps for you:

1. **Generate Plots**: Use your `running_results_exp_114.csv` to create a bar chart comparing "Old Triplet" vs "New MS-Loss".
2. **Qualitative Visualization**: Pick 3 images from your `ood_inspection` folder—one correctly predicted but with low confidence, and one incorrect prediction—to include in Section 5.

Would you like a Python snippet to automatically generate these comparison plots from your results CSV?