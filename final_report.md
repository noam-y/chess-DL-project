# Chessboard State Prediction from a Single RGB Image

## 1. Abstract

This project addresses automatic chessboard state prediction from a single RGB image by classifying all 64 squares and reconstructing board state. We developed and compared multiple pipelines, starting from a patch-based CNN baseline and progressing to a configurable multi-head ResNet-18 model with metric-learning regularization and fold-wise ensembling. The final training protocol uses game-based cross-validation, weighted sampling to reduce class imbalance, and a held-out unseen game for final testing. Our primary metric is macro F1, chosen to prevent inflated performance from dominant empty-square classes. The resulting framework is robust, reproducible, and aligned with the official course submission requirements.

## 2. Introduction

### Task Description and Motivation

The goal is to convert a chessboard image into a structured board-state representation by predicting one class per square. This is a practical computer-vision problem for automated game digitization, analysis, and broadcasting.

### Challenges and Goals

The task is difficult because:

- empty squares are much more frequent than piece squares (severe class imbalance),
- piece appearances vary across games, boards, lighting, and camera viewpoints,
- several classes are visually similar (for example bishop vs pawn under blur or occlusion),
- errors in a few squares can make the entire board reconstruction wrong.

Our goal is to improve class-balanced recognition and cross-game generalization, not just average accuracy on easy/majority classes.

### Main Contributions

- A configurable multi-head ResNet-18 architecture that decomposes board-square classification into structured sub-decisions.
- A full game-based cross-validation protocol with a strict unseen test game.
- Comparative metric-learning experiments (hard triplet style vs multi-similarity style regularization).
- Ensemble inference across fold-best checkpoints to reduce variance.
- Practical OOD/uncertainty inspection tooling through confidence-threshold analysis.

## 3. Related Work

This project builds on three relevant directions:

1. **CNN classification for local image patches**, used in our early baseline (`train.py`).
2. **Residual transfer learning (ResNet-18)**, used as the main backbone in the advanced pipeline (`experiment114.py`, `experiment2.py`, `SUPER_SEARCH_9001.py`).
3. **Metric learning for embedding separation**, including triplet-style and multi-similarity losses.

Compared with end-to-end board detectors or transformer pipelines, our approach emphasizes controlled per-square classification with explicit label structure (occupancy and piece identity decomposition), which matches the available labels and training organization in this repository.

## 4. Method

### 4.1 Input and Output Representation

Training labels are derived from FEN tokens at square level:

- `e` for empty
- uppercase letters for white pieces
- lowercase letters for black pieces

During training, labels are mapped into a unified 13-class space (empty + 12 pieces). At inference, predictions are converted back into board notation (FEN-like row serialization) for board-level evaluation.

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
- **2-head**: occupancy (2-class) + piece identity (12-class).
- **3-head**:
  - variant A: occupancy + color + 6-way piece type,
  - variant B (latest branch): occupancy + white-piece head + black-piece head.

This decomposition helps separate easy/global decisions (empty vs occupied) from harder fine-grained piece classification.

### 4.4 Training Procedure

Core configuration used in the focused experiment branch:

- optimizer: Adam (`lr=1e-4`, `weight_decay=1e-4`)
- scheduler: `ReduceLROnPlateau` on validation macro F1
- early stopping after sustained non-improvement
- up to 100 epochs per fold (higher in some exploratory scripts)

A robust custom collator skips corrupted samples during batch assembly.

### 4.5 Loss Functions

Total loss combines classification and optional metric regularization:

- cross-entropy for occupancy and piece heads,
- plus one metric term:
  - **old**: hard triplet-style margin loss,
  - **new**: multi-similarity loss on normalized embeddings.

Metric terms are applied only on occupied-square subsets (and color-conditional subsets when relevant).

### 4.6 Class Imbalance Handling

We use `WeightedRandomSampler` with a `50_50` mode that balances empty vs non-empty mass. This increases learning signal for rare piece classes and improves macro-oriented metrics.

### 4.7 Ensemble Inference

For each configuration, fold-best checkpoints are loaded and their probabilities are averaged on the unseen test game. For multi-head variants, head outputs are fused into unified 13-class probabilities before averaging and argmax.

## 5. Experiments

### 5.1 Experimental Setup

To cover the full search space explored across repository experiment scripts, we define the experiment dimensions as:

- **Heads**: `1`, `2`, `3a`, `3b`
  - `1`: unified 13-class head
  - `2`: occupancy + 12-piece head
  - `3a`: occupancy + color + shared 6-piece head
  - `3b`: occupancy + white 6-piece head + black 6-piece head
- **Sampling**: `50_50`, `none`, `uniform`
- **Metric mode**: `none`, `old`, `new`
  - `none`: cross-entropy only
  - `old`: triplet-style metric regularization
  - `new`: multi-similarity metric regularization
- **Batch size**: `16`, `32`, `64`
- **Freeze strategy**: `none`, `old`, `new`
  - `none`: backbone trainable
  - `old`: static backbone freezing
  - `new`: progressive unfreezing schedule

This section intentionally reflects the complete experimental intent (not only a narrow subset in one script version).

### 5.2 Evaluation Metrics

Primary metric:

- **Macro F1 (%)** at square level (class-balanced)
  - F1 is a standard machine-learning metric (not a custom metric in this project), defined as: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`.
  - We treat each square prediction as one classification sample, then compute F1 separately for each class (empty + 12 piece classes).
  - Macro F1 is the unweighted mean of those per-class F1 values, so every class contributes equally.
  - This is important because empty squares are much more frequent than piece classes; plain accuracy can look high even when rare pieces are predicted poorly.
  - Reporting in percent is simply: `Macro F1 * 100`.

Secondary metrics (from baseline evaluation utilities):

- piece accuracy (correct square labels over all squares)
- board accuracy (exact 64/64 board reconstruction)

### 5.3 Quantitative Results

The experiment pipelines are designed to export running and final CSV summaries (for example: `running_results.csv`, `FINAL_RESULTS.csv`, `running_results_exp.csv`, `running_results_exp_114.csv`, `FINAL_RESULTS_114.csv`, and per-task files in parallel mode).

At the time of writing, final consolidated CSV artifacts are not committed in this branch, so numeric tables should be filled from the latest completed run outputs.

Planned final table schema:

| Config ID | Heads | Sampling | Metric | Batch | Freeze | Mean 4-Fold F1 (%) | Unseen Test Ensemble F1 (game5, %) |
|---|---|---|---|---:|---|---:|---:|
| Example_Config_1 | 1 | 50_50 | old | 32 | old | TBD | TBD |
| Example_Config_2 | 2 | uniform | new | 16 | new | TBD | TBD |
| Example_Config_3 | 3a | none | none | 64 | none | TBD | TBD |
| Example_Config_4 | 3b | 50_50 | new | 32 | new | TBD | TBD |

### 5.4 Qualitative Results

Qualitative analysis is performed via confidence-based inspection:

- script: `inference_using_kfold.py`
- outputs:
  - `low_confidence_images.csv`
  - `ood_inspection/` (copied low-confidence examples)

These samples should be included as figures in the final PDF to illustrate difficult cases such as edge crops, partial occlusions, and visually ambiguous squares.

## 6. Ablation Study (Required)

Our method contains multiple components, so we evaluate each one by controlled comparison.

### 6.1 Components

1. **Head design**: `1` vs `2` vs `3a` vs `3b`
2. **Metric loss**: `none` vs `old` vs `new`
3. **Sampling strategy**: `none` vs `uniform` vs `50_50`
4. **Batch size**: `16` vs `32` vs `64`
5. **Freeze strategy**: `none` vs `old` vs `new`

### 6.2 Ablation Questions

- Does hierarchical head design (`2`, `3a`, `3b`) outperform a single unified head (`1`) on macro F1?
- Does metric regularization (`old` or `new`) outperform no metric term (`none`)?
- Which sampler (`none`, `uniform`, `50_50`) best handles class imbalance without harming overall generalization?
- How sensitive is performance to batch size (`16`, `32`, `64`)?
- Which freeze regime (`none`, `old`, `new`) yields the best stability and final unseen-game performance?

### 6.3 Expected Interpretation

- Gains from `3a/3b` over `1/2` indicate deeper task decomposition improves difficult class decisions.
- Gains from `new` over `old`/`none` indicate improved embedding geometry for hard negatives.
- Gains from `50_50` over `none`/`uniform` indicate better minority-class handling.
- Best performance at a specific batch size reflects optimization/generalization trade-offs.
- Freeze strategy differences reflect transfer-learning stability versus adaptation flexibility.

## 7. What Did Not Work (Optional)

- The initial patch-CNN baseline (`train.py`) was fast to prototype but underpowered for hard generalization.
- Large broad grid scripts accumulated code complexity and branch divergence, making it harder to maintain one clean experiment source of truth.
- Parallel SLURM execution required per-task result files to avoid shared CSV overwrite/race conditions.
- Some exploratory scripts contain logic bugs or inconsistent variables, so final reporting should rely on the stable script path.

## 8. Discussion / Limitations (Optional but Recommended)

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
- Evaluate stronger backbones (for example lightweight ViT variants) under the same split protocol.

## 9. References

1. He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016.
2. Schroff, F., Kalenichenko, D., Philbin, J. FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR, 2015.
3. Wang, X., Han, X., Huang, W., Dong, D., Scott, M. R. Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning. CVPR, 2019.
4. PyTorch Documentation. https://pytorch.org

---

## Reproducibility Notes (Appendix)

- Main run command: `python experiment114.py`
- OOD inspection: `python inference_using_kfold.py --models_path <path> --heads <1|2|3> --dataset_path assets/new_dataset --threshold 0.5`
- Baseline training/evaluation scripts are available in `train.py`, `train_v2.py`, `evaluate.py`, and `evaluate_v2.py`.
