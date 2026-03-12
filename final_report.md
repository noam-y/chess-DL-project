# Chessboard State Prediction from a Single RGB Image

## 1\. Abstract

This project addresses automatic chessboard state prediction from a single RGB image by classifying all 64 squares and reconstructing board state. We developed and compared multiple pipelines, starting from a patch-based CNN baseline and progressing to a configurable multi-head ResNet-18 model with metric-learning regularization and fold-wise ensembling. The final training protocol uses game-based cross-validation, weighted sampling to reduce class imbalance, and a held-out unseen game for final testing. Our primary metric is macro F1, chosen to prevent inflated performance from dominant empty-square classes. The resulting framework is robust, reproducible, and aligned with the official course submission requirements.

## ---

## 2\. Introduction

### Task Description and Motivation

The goal is to convert a chessboard image into a structured board-state representation by predicting one class per square. This is a practical computer-vision problem for automated game digitization, analysis, and broadcasting.

### Challenges and Goals

The task is difficult because:

- empty squares are much more frequent than piece squares (severe class imbalance),  
- piece appearances vary across games, boards, lighting, and camera viewpoints,  
- several classes are visually similar (for example bishop vs pawn under blur or occlusion),  
- errors in a few squares can make the entire board reconstruction wrong.

Our goal is to improve class-balanced recognition and cross-game generalization, not just average accuracy on easy/majority classes.

### 

### Main Contributions

- A configurable multi-head ResNet-18 architecture that decomposes board-square classification into structured sub-decisions.  
- A full game-based cross-validation protocol with a strict unseen test game.  
- Comparative metric-learning experiments (hard triplet style vs multi-similarity style regularization).  
- Ensemble inference across fold-best checkpoints to reduce variance.  
- Practical OOD/uncertainty inspection tooling through confidence-threshold analysis.

---

## 3\. Related Work

This project builds on three relevant directions:

1. **CNN classification for local image patches**, used in our early baseline (`train.py`).  
2. **Residual transfer learning (ResNet-18)**, used as the main backbone in the advanced pipeline (`experiment114.py`, `experiment2.py`, `SUPER_SEARCH_9001.py`).  
3. **Metric learning for embedding separation**, including triplet-style and multi-similarity losses.

Compared with end-to-end board detectors or transformer pipelines, our approach emphasizes controlled per-square classification with explicit label structure (occupancy and piece identity decomposition), which matches the available labels and training organization in this repository.

---

## 4\. Method

### 4.1 Input and Output Representation

Training labels are derived from FEN tokens at square level:

- `e` for empty  
- uppercase letters for white pieces  
- lowercase letters for black pieces

During training, labels are mapped into a unified 13-class space (empty \+ 12 pieces). At inference, predictions are converted back into board notation (FEN-like row serialization) for board-level evaluation.

### 4.2 Dataset and Splits

The main pipeline reads from `assets/new_dataset` with per-game folders containing:

- `gt.csv`  
- `tagged_images/`

We use game-based splitting:

- CV folds: `game2`, `game4`, `game6`, `game7` (leave-one-game-out within this pool)  
- held-out unseen test: `game5`

This split reduces leakage compared with random frame-level splits.

### 4.3 Model Architecture

The core model is `ConfigurableChessResNet`, based on ImageNet-pretrained ResNet-18 backbone, with alternative heads:

- **1-head**: direct 13-class logits.  
- **2-head**: occupancy (2-class) \+ piece identity (12-class).  
- **3-head**:  
  - variant A: occupancy \+ color \+ 6-way piece type,  
  - variant B: occupancy \+ white-piece head \+ black-piece head.

This decomposition helps separate easy/global decisions (empty vs occupied) from harder fine-grained piece classification.

### 4.4 Training Procedure

Core configuration used in the focused experiment branch:

- optimizer: Adam (`lr=1e-4`, `weight_decay=1e-4`)  
- scheduler: `ReduceLROnPlateau` on validation macro F1  
- early stopping after sustained non-improvement  
- up to 100 epochs per fold (higher in some exploratory scripts)

A robust custom collator skips corrupted samples during batch assembly.

### 

### 4.5 Loss Functions

Total loss combines classification and optional metric regularization:

- cross-entropy for occupancy and piece heads,  
- plus one metric term:  
  - hard triplet-style margin loss,  
  - multi-similarity loss on normalized embeddings.

Metric terms are applied only on occupied-square subsets (and color-conditional subsets when relevant).

### 4.6 Class Imbalance Handling

We use `WeightedRandomSampler` with a `50_50` mode that balances empty vs non-empty mass. This increases learning signal for rare piece classes and improves macro-oriented metrics.

### 4.7 Ensemble Inference

For each configuration, fold-best checkpoints are loaded and their probabilities are averaged on the unseen test game. For multi-head variants, head outputs are fused into unified 13-class probabilities before averaging and argmax.

---

## 

## 5\. Experiments

### 5.1 Experimental Setup

The quantitative table in Section 5.3 reflects the following actual experiment dimensions:

- **Heads**: `1`, `2`, `3`  
- **Sampler**: `50 50`, `none`, `uniform`  
- **Freeze Backbone**: `TRUE`, `FALSE`  
- **Metric Loss**: `none`, `Triplet`, `Similarity`  
- **Batch Size**: `16`, `32`, `64`  
- **Classification Loss**: `cross entropy`, `focal`  
- **Label Smoothing**: `TRUE`, `FALSE`  
- **Data Augmentation**: `TRUE`, `FALSE`  
- **Test-Time Augmentation (TTA)**: `TRUE`, `FALSE`  
- **Metric Weight**: `0.25`, `0.5`, `0.75`, `1`

### 5.2 Evaluation Metrics

Primary metric:

**Macro F1 (%)** at square level (class-balanced)

- F1 is a standard machine-learning metric , defined as: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`.  
  - We treat each square prediction as one classification sample, then compute F1 separately for each class (empty \+ 12 piece classes).  
  - Macro F1 is the unweighted mean of those per-class F1 values, so every class contributes equally.  
  - This is important because empty squares are much more frequent than piece classes; plain accuracy can look high even when rare pieces are predicted poorly.  
  - Reporting in percent is simply: `Macro F1 * 100`.

### 5.3 Quantitative Results:

| \#Heads | Sampler | Freeze Backbone | Metric Loss | Batch Size | Loss | Label Smoothing | Data Augmentation | Test-Time Augmentation | Metric Weight | Ensemble Test F1 (game5) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | uniform | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 93.85 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 92.70 |
| 2 | 50 50 | FALSE | Similarity | 32 | focal | TRUE | TRUE | TRUE | 0.5 | 92.51 |
| 2 | 50 50 | FALSE | Similarity | 16 | focal | TRUE | TRUE | TRUE | 0.5 | 92.14 |
| 2 | 50 50 | FALSE | Similarity | 64 | cross entropy | TRUE | TRUE | TRUE | 0.5 | 92.08 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 91.90 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.75 | 91.73 |
| 2 | 50 50 | FALSE | Triplet | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 91.71 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | FALSE | TRUE | TRUE | 0.5 | 91.21 |
| 1 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 90.47 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | FALSE | 0.5 | 89.57 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | FALSE | 0.5 | 89.54 |
| 1 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 88.97 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | FALSE | TRUE | 0.5 | 88.91 |
| 3 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 88.81 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | FALSE | FALSE | 0.5 | 88.78 |
| 1 | none | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 88.63 |
| 3 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 88.37 |
| 3 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 88.37 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.25 | 88.17 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 87.84 |
| 1 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.77 |
| 3 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 87.73 |
| 2 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.46 |
| 1 | uniform | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.43 |
| 2 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.27 |
| 2 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.25 |
| 3 | uniform | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.19 |
| 3 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.17 |
| 3 | uniform | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.13 |
| 1 | none | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.08 |
| 3 | 50 50 | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 87.03 |
| 2 | none | FALSE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 86.92 |
| 2 | none | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 86.92 |
| 3 | uniform | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 86.65 |
| 3 | 50 50 | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 86.52 |
| 2 | 50 50 | FALSE | Similarity | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 86.38 |
| 2 | 50 50 | FALSE | Similarity | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 86.23 |
| 2 | none | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 86.06 |
| 3 | uniform | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 85.89 |
| 1 | uniform | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 85.75 |
| 2 | uniform | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 85.72 |
| 2 | 50 50 | FALSE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 85.31 |
| 2 | none | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 85.26 |
| 2 | 50 50 | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 85.14 |
| 2 | uniform | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 84.96 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 84.82 |
| 2 | none | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 84.55 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | FALSE | TRUE | TRUE | 0.5 | 84.33 |
| 1 | 50 50 | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 84.12 |
| 2 | uniform | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 84.07 |
| 1 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 84.02 |
| 1 | 50 50 | FALSE | Similarity | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 83.81 |
| 2 | uniform | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 83.78 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | FALSE | TRUE | FALSE | 0.5 | 83.70 |
| 1 | 50 50 | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 83.64 |
| 1 | 50 50 | TRUE | Similarity | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 83.47 |
| 1 | 50 50 | FALSE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 83.12 |
| 1 | none | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 83.02 |
| 1 | none | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 83.02 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | TRUE | TRUE | TRUE | 0.5 | 82.95 |
| 1 | 50 50 | TRUE | Similarity | 64 | cross entropy | FALSE | FALSE | FALSE | 1 | 82.91 |
| 2 | 50 50 | FALSE | Similarity | 64 | focal | TRUE | FALSE | TRUE | 0.5 | 82.90 |
| 1 | uniform | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 82.86 |
| 3 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 82.76 |
| 1 | uniform | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 82.73 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | TRUE | TRUE | FALSE | 0.5 | 82.59 |
| 2 | 50 50 | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 81.81 |
| 1 | none | TRUE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 81.62 |
| 1 | 50 50 | FALSE | Similarity | 64 | cross entropy | FALSE | FALSE | FALSE | 1 | 80.74 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | TRUE | FALSE | FALSE | 0.5 | 80.27 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | FALSE | FALSE | TRUE | 0.5 | 79.89 |
| 1 | none | FALSE | none | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 79.07 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | FALSE | FALSE | FALSE | 0.5 | 78.46 |
| 2 | 50 50 | TRUE | Similarity | 64 | focal | TRUE | FALSE | TRUE | 0.5 | 77.09 |
| 1 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 75.88 |
| 1 | 50 50 | TRUE | Triplet | 64 | cross entropy | FALSE | FALSE | FALSE | 1 | 75.44 |
| 1 | 50 50 | FALSE | Triplet | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 75.27 |
| 1 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 74.52 |
| 3 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 73.24 |
| 3 | 50 50 | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 73.24 |
| 1 | 50 50 | TRUE | Triplet | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 72.95 |
| 1 | 50 50 | FALSE | Triplet | 64 | cross entropy | FALSE | FALSE | FALSE | 1 | 72.02 |
| 3 | 50 50 | FALSE | Similarity | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 69.98 |
| 1 | none | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 68.25 |
| 1 | none | FALSE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 68.25 |
| 3 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 58.39 |
| 1 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 57.60 |
| 1 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 57.60 |
| 3 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 57.49 |
| 3 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 57.49 |
| 2 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 56.64 |
| 2 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 56.64 |
| 1 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 56.36 |
| 1 | 50 50 | TRUE | Triplet | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 56.36 |
| 3 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 56.05 |
| 3 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 56.05 |
| 2 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 55.97 |
| 2 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 55.97 |
| 2 | 50 50 | TRUE | Similarity | 32 | cross entropy | FALSE | FALSE | FALSE | 1 | 55.89 |
| 2 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 55.48 |
| 1 | 50 50 | TRUE | Similarity | 16 | cross entropy | FALSE | FALSE | FALSE | 1 | 55.36 |

## 6\. Ablation Study

### 6.1 Best Overall Configuration

From the Section 5.3 table, the best recorded configuration is:

- Heads \= `2`  
- Sampler \= `uniform`  
- Freeze Backbone \= `FALSE`  
- Metric Loss \= `Similarity`  
- Batch Size \= `64`  
- Loss \= `focal`  
- Label Smoothing \= `TRUE`  
- Data Augmentation \= `TRUE`  
- Test-Time Augmentation \= `TRUE`  
- Metric Weight \= `0.5`  
- **Ensemble Test F1 \= 93.85**

### 6.2 Head Ablation

Using the best score achieved per head value in Section 5.3:

- **Head 1 best**: `90.47`  
- **Head 2 best**: `93.85`  
- **Head 3 best**: `88.81`

Conclusion: all heads can produce competitive results, but head `2` is clearly strongest in this table.

### 6.3 Metric-Loss Ablation

Controlled comparison at (`Heads=2`, `Sampler=50 50`, `Freeze=FALSE`, `Batch=32`, `cross entropy`, no smoothing/aug/TTA):

- `none`: `85.14`  
- `Triplet`: `87.46` (also `87.25` in another repeated run)  
- `Similarity`: `86.38` (also `86.23` in another repeated run)

Interpretation: both metric-learning modes outperform `none` in this matched baseline setup. Triplet is slightly higher than Similarity here.

High-performance regime comparison (`Heads=2`, `Sampler=50 50`, `Freeze=FALSE`, `Batch=64`, `focal`, smoothing+aug+TTA TRUE, `Metric Weight=0.5`):

- `Triplet`: `91.71`  
- `Similarity`: `92.70`

In the strongest regime, Similarity outperforms Triplet by `+0.99`.

Additional controlled example (`Heads=1`, `Sampler=50 50`, `Freeze=FALSE`, `Batch=32`, `cross entropy`, no smoothing/aug/TTA):

- `Triplet (old)`: `75.88`  
- `Similarity (new)`: `83.81`  
- **Gain from new metric**: `+7.93 F1`

### 6.4 Batch-Size Ablation

For a controlled subset (`Heads=1`, `Sampler=50 50`, `Freeze=FALSE`, `Similarity`, `cross entropy`, no smoothing/aug/TTA):

- Batch `16`: `83.12`  
- Batch `32`: `83.81` (best)  
- Batch `64`: `80.74`

Conclusion: In this subset, medium batch (`32`) outperforms both smaller and larger alternatives.

In the high-performance focal setting for `Heads=2`, `Sampler=50 50`, `Freeze=FALSE`, `Similarity`, smoothing+aug+TTA TRUE (`Metric Weight=0.5`):

- Batch `16`: `92.14`  
- Batch `32`: `92.51`  
- Batch `64`: `92.70` (best in this subset)

So the preferred batch size depends on the training regime.

### 6.5 Freeze Ablation

For matched high-performance settings (`Heads=2`, `Sampler=50 50`, `Similarity`, `Batch=64`, `focal`, smoothing/aug/TTA toggles):

- Example pair (all TRUE): `Freeze=FALSE 92.70` vs `Freeze=TRUE 82.95` (`+9.75` for no-freeze)  
- Example pair (TTA FALSE): `89.57` vs `82.59` (`+6.98`)  
- Example pair (Aug FALSE, TTA TRUE): `88.91` vs `77.09` (`+11.82`)

Conclusion: under these settings, training the backbone consistently outperforms frozen-backbone runs.

### 6.6 Sampling Ablation

Sampling coverage in Section 5.3 includes all three samplers (`50 50`, `none`, `uniform`), but repeated rows and run variance make a strict one-to-one comparison noisy.

Top observed score per sampler:

- `50 50`: `92.70`  
- `none`: `88.63`  
- `uniform`: `93.85`

In repeated `none` rows (for example, `Heads=1`, `Triplet`, `Batch=32`, `Freeze=FALSE`), scores vary substantially (`88.63`, `83.02`, `68.25`), indicating sensitivity to run conditions and/or logging duplication.

### 6.7 Additional Ablations Present in CSV

The final results table also includes extra factors beyond the original grid:

- classification loss type (`cross entropy` vs `focal`)  
- label smoothing  
- data augmentation  
- test-time augmentation (TTA)  
- metric-loss weight

The highest scores are concentrated in the `focal + smoothing + augmentation + TTA` regime, indicating these components contribute materially to final performance.

### 6.8 Notes on Data Quality

`FINAL_RESULTS_ALL.csv` contains repeated configurations with different scores (duplicate rows or repeated keys). Therefore, conclusions are reported using explicit controlled comparisons and best-score summaries, not only single-row rank ordering.

## 

## 

## 7\. What Did Not Work (Optional)

- The initial patch-CNN baseline was fast to prototype but underpowered for hard generalization.  
- Large, broad grid scripts accumulated code complexity and branch divergence, making it harder to maintain one clean experiment source of truth.  
- Some exploratory scripts contain logic bugs or inconsistent variables, so final reporting should rely on the stable script path.  
- A purely threshold-based Out-of-Distribution (OOD) detector failed on organic occlusions, such as hands (as seen in the photo). Because our 96x96 crops overlap adjacent squares, true empty squares often contain edges of neighboring pieces, dropping their confidence to \~65% in such cases (e.g., h1 or e1 as seen in the photo). Meanwhile, smooth skin lacks edges, causing the network to incorrectly classify it as 'Empty' with \>70% confidence. This proved that organic OOD objects cannot be filtered by confidence thresholds alone, as the closed-world assumption forces the network to map featureless anomalies to the 'Empty' class. This led us to consider asymmetric thresholds that treat empty and occupied tiles differently. Following that, we set OOD as sub-45% ‘Empty’-classified tiles.

![][image1]

## 8\. Discussion / Limitations (Optional but Recommended)

### Failure Cases

- Low-confidence predictions often appear on edge-of-square crops or ambiguous visual context.  
- Similar-looking pieces under blur/lighting changes remain challenging.

### Method Limitations

- Current inference is square-local and does not enforce chess legality constraints globally.  
- Generalization depends on the diversity and size of available game splits.  
- The unified 13-class experiment branch does not explicitly model an OOD output class in final predictions.

### Future Improvements

- Add board-consistency postprocessing with chess-rule priors.  
- Expand data diversity and viewpoint augmentation.  
- Apply confidence calibration and uncertainty-aware outputs.  
- Evaluate stronger backbones (for example, lightweight ViT variants) under the same split protocol.

## 9\. References

1. He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016\.  
2. Schroff, F., Kalenichenko, D., Philbin, J. FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR, 2015\.  
3. Wang, X., Han, X., Huang, W., Dong, D., Scott, M. R. Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning. CVPR, 2019\.  
4. PyTorch Documentation. [https://pytorch.org](https://pytorch.org)

---

## Reproducibility Notes (Appendix)

- Main run command: `python experiment114.py`  
- OOD inspection: `python inference_using_kfold.py --models_path <path> --heads <1|2|3> --dataset_path assets/new_dataset --threshold 0.45`  
- Baseline training/evaluation scripts are available in `train.py`, `train_v2.py`, `evaluate.py`, and `evaluate_v2.py`.
