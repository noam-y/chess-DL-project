This report structure reflects the standard research and development workflow used on the **BGU ISE-CS-DT GPU Cluster**. Based on your project files and the cluster's operational guidelines, here is the context needed to populate each section.

---

## 1. Abstract

* **Problem:** Automating chess board state recognition from images.
* **Approach:** A deep learning pipeline using a **ResNet18 backbone** to classify $96 \times 96$ tiles extracted from $480 \times 480$ board images.
* **Main Results:** Evaluation of 36 model permutations (Heads, Sampling, Triplet Loss, Freezing) using **4-Fold Cross-Validation** and an ensemble test on an unseen game.

## 2. Introduction

* **Task:** Converting raw images of chess boards into formal **FEN (Forsyth-Edwards Notation)** strings.
* **Motivation:** Facilitating automated game analysis and digital record-keeping from video frames.
* **Challenges:** Handling overlapping pieces, varying lighting, and maintaining spatial context between tiles.

## 3. Related Work

* **Datasets:** Custom-built dataset organized by game folders (`game2`, `game4`, etc.) containing `gt.csv` and `tagged_images`.
* **Repositories:** Developed and maintained on a private Git repository with a `protocol` branch.
* **Prior Work:** Built upon standard ResNet architectures for feature extraction.

## 4. Method

* **Model Architecture:** `ConfigurableChessResNet` using a ResNet18 backbone. It supports three head configurations:
1. **Unified:** 13-class classifier (12 pieces + 1 empty).
2. **Two-Head:** Occupancy (2) and Piece Type (12).
3. **Three-Head:** Occupancy (2), Color (2), and Piece Type (6).


* **Input Representation:** $96 \times 96$ RGB tiles with 50% padding ($192 \times 192$ extraction area) to capture pieces crossing tile boundaries.
* **Preprocessing:** Fixed geometric partitioning of $480 \times 480$ images into $60 \times 60$ squares, expanded to $96 \times 96$ tiles.
* **Loss Functions:** Cross-Entropy Loss combined with an optional **Hard-Triplet Loss** for feature clustering.

## 5. Experiments

* **Dataset Splits:** 4-Fold Cross-Validation using `game2`, `game4`, `game6`, and `game7` for training/validation. `game5` is reserved as the final unseen test set.
* **Metrics:** **Macro F1-Score** (to account for class imbalance between empty squares and rare pieces).
* 
**Hardware:** Conducted on the BGU GPU Cluster using **NVIDIA RTX 3090 GPUs**.


* **Training Configuration:** Adam optimizer ($lr=0.0001$), weight decay ($1e-4$), and **Early Stopping** with a patience of 5 epochs.

## 6. Ablation Study

* **Component 1 (Backbone Freezing):** Comparison between fixed ImageNet features vs. full fine-tuning.
* **Component 2 (Multi-Head vs. Unified):** Analyzing if breaking the task into sub-tasks (Occupancy/Color) improves F1-scores.
* **Component 3 (Triplet Loss):** Measuring the impact of forced feature separation on classification accuracy.
* **Component 4 (Sampling):** Comparing 'Uniform' and '50/50' sampling against 'None' to address the high frequency of empty ('e') tiles.

## 7. What Did Not Work

* **Initial Time Constraints:** Early attempts failed due to the **4-hour Slurm timeout**.
* 
**Memory Issues:** Initial large batch sizes caused **CUDA Out of Memory (OOM)** errors, necessitating a reduction to batch size 32.


* **Backbone Selection:** Heavier models (e.g., ResNet50) were abandoned in favor of ResNet18 for faster grid search iteration.

## 8. Discussion / Limitations

* **Failure Cases:** Misclassification of pieces that are partially obscured or located at the extreme edges of the board.
* **Limitations:** The current method assumes the board is already cropped to a $480 \times 480$ square; it does not yet handle perspective distortion from angled camera views.
* 
**Future Work:** Implementing a "Golden Card" priority system for longer training runs or using **Job Arrays** to scale hyperparameter searches.



---

**Next Step:** Would you like me to generate the **Quantitative Results** tables based on the `running_results.csv` file produced by your grid search?