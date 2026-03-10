# Complete Project Context (Consolidated)

This file consolidates all available context from:

- `Projects Final Submission .pdf`
- `agent1_context.md`
- `agent2_context.md`
- `agent3_context.md`
- `agent4_context.md`
- `final_report.md`
- All repository code/docs/scripts (excluding `.git` internals)


## 2) Project Scope and Objective

The project is a chessboard state recognition system focused on classifying board squares from image data and reconstructing board state (FEN-like representation). The core direction evolved from:

1. Early full-board patch-based CNN (`train.py` + `evaluate.py`)
2. Improved contextual patch + ResNet workflow (`train_v2.py` + `evaluate_v2.py`)
3. Main research branch: configurable multi-head ResNet-18 with CV-based grid search, metric learning, and ensemble evaluation (`SUPER_SEARCH_9001.py`, `experiment2.py`, `parallel_experiment.py`, `experiment114.py`)

Primary target metric in the main branch is macro F1 at square level, with game-wise split protocol and unseen held-out test game.

---

## 3) Submission/Evaluation Requirements from PDF

`Projects Final Submission .pdf` defines:

- Dataset format requirements (images + `gt.csv` conventions)
- Final report scientific-paper structure:
  1. Abstract
  2. Introduction
  3. Related Work
  4. Method
  5. Experiments
  6. Ablation (required)
  7. What did not work (optional)
  8. Discussion/limitations
  9. References
- Code submission requirements:
  - GitHub repo
  - `requirements.txt`
  - reproducible instructions in `README.md`
- Evaluation API requirement for chess projects:
  - `predict_board(image: np.ndarray) -> torch.Tensor` with strict output format/encoding
  - class encoding includes empty and optional OOD (project-dependent)

Important practical metadata in the PDF:

- Presentation and submission date guidance
- Emphasis on clarity, ablation evidence, and reproducibility

---

## 4) Context Files Synthesis

### `agent1_context.md`

- Strong draft for final report around:
  - Multi-head ResNet-18
  - Multi-Similarity loss improvement over older triplet setups
  - 4-fold game-based validation + game5 unseen test
  - Reported best ensemble F1 around ~84% (in this draft)

### `agent2_context.md`

- Cluster-oriented report framing:
  - BGU SLURM workflow
  - hardware constraints (timeouts, OOM, GPU quotas)
  - architecture and split description similar to core codebase narrative

### `agent3_context.md`

- Alternative framing with higher reported accuracy (~94.74% LOGO mean accuracy in this draft)
- Emphasizes masked hierarchical CE and domain generalization
- Includes methodological narrative that partially overlaps but does not fully match latest scripts

### `agent4_context.md`

- Broad, detailed project writeup mapped to formal report sections
- Explicitly notes that final quantitative tables may still require inserting outputs from experiment CSVs
- Aligns closely with `experiment114.py` branch and mentions ensemble outputs + metric-learning comparisons

### `final_report.md`

- Most complete integrated report draft currently in repo
- Covers method, experiments, reproducibility, and references
- Keeps some placeholders/TBD-like parts where final numeric artifacts are needed

### Key observation across context files

There are multiple report narratives with overlapping structure but conflicting specific numbers/config details. The underlying code history suggests iterative experimentation; not all draft claims can be simultaneously true without selecting one definitive experiment branch and final results table.

---

## 5) Repository Codebase Map (Non-`.git`)

### Core experiment scripts

- `SUPER_SEARCH_9001.py`
  - 36-combination grid search
  - knobs: head count (2/3), sampling (`uniform`/`50_50`/`none`), triplet on/off, freeze on/off
  - 4-fold (game2, game4, game6, game7), ensemble eval on game5
  - outputs under `SUPER_SEARCH_9001_results/`

- `experiment2.py`
  - expanded framework with:
    - optional progressive unfreezing
    - multi-similarity loss implementation (`calculate_multi_similarity_loss`)
    - selected combo list + intended 72-combination framing
  - contains a critical bug:
    - `todo = combs.append(itertools.product(...))` makes `todo` become `None`

- `parallel_experiment.py`
  - array-job version for cluster parallelism
  - uses `SLURM_ARRAY_TASK_ID` to execute exactly one combination
  - writes per-task CSV (`results_task_<id>.csv`) to avoid shared write conflicts

- `experiment114.py`
  - focused configuration sweep:
    - heads: 2, 3
    - sampling: `50_50`
    - metric mode: `new`/`old`
    - batch size: 32
  - explicit white/black split heads for 3-head variant
  - fold checkpoints + ensemble test + running/final CSV export
  - currently appears to be the cleanest/latest experiment script

- `temp.py`
  - branch-like copy with mixed edits
  - includes inconsistent variables (`use_triplet`, `epoch_ce_loss_sum`, `epoch_batches`) that are not fully defined in shown flow
  - likely scratch or intermediate file

### Inference/OOD utility

- `inference_using_kfold.py`
  - loads ensemble checkpoints
  - computes max confidence per sample
  - exports low-confidence samples to:
    - `low_confidence_images.csv`
    - `ood_inspection/`

### Earlier baseline training/eval

- `train.py`
  - baseline patch CNN (`PieceClassifier`)
  - full board resized to 480x480, split to 60x60 patches
  - weighted CE (lower empty-class weight)

- `evaluate.py`
  - evaluates `train.py` model
  - computes piece-level and board-perfect accuracy
  - writes `evaluation_results.csv`

### Improved contextual baseline (v2)

- `train_v2.py`
  - `SmartChessDataset` with contextual crop (1.5x square) and stronger augmentations
  - ResNet-18 based classifier
  - hold-out game validation setup

- `evaluate_v2.py`
  - robust image-path resolution logic
  - contextual crop inference pipeline matching `train_v2.py`
  - computes piece and board accuracy

### Data augmentation utility

- `board_generator.py`
  - patch mixing synthetic augmentation pipeline
  - reads labels CSV, matches frames, recombines board patches, outputs:
    - generated images
    - `augmented_ground_truth.csv`

### Documentation and ops scripts

- `readme.md`
  - cluster setup notes, environment setup, training/eval commands
- `Instructions.md`
  - short launcher for master grid search
- `requirements.txt`
  - currently lists: `torch`, `torchvision`, `numpy`, `dandas`, `Pillow`, `tqdm`
  - likely typo: `dandas` should be `pandas`
- `.gitignore`
  - ignores `assets/`, experiment outputs, `.pth`, logs, caches
- SLURM scripts:
  - `master_grid.sh`
  - `sbatch_experiment2.sh`
  - `sbatch_experiment114.sh`
  - `sbatch_parallel.sh`
- `deploy.sh`
  - auto commit/push + remote ssh submission helper

---

## 6) Data and Split Conventions

Across primary experiment scripts, expected dataset layout is:

- `assets/new_dataset/<game>/gt.csv`
- `assets/new_dataset/<game>/tagged_images/...`

Common split logic:

- Validation folds: `game2`, `game4`, `game6`, `game7`
- Held-out test: `game5`

Label conventions:

- Unified 13-class mapping:
  - empty + 12 chess piece classes
- Additional helper mappings:
  - 12-piece collapsed labels
  - 6-piece type labels for color-conditional heads

---

## 7) Modeling and Loss Evolution

### Architecture progression

1. Single-head patch CNN baseline (`train.py`)
2. ResNet patch classifier with contextual crops (`train_v2.py`)
3. Configurable multi-head ResNet-18:
   - 1-head: direct 13-class
   - 2-head: occupancy + 12-piece
   - 3-head variants:
     - occupancy + color + 6-piece
     - occupancy + white-head + black-head (latest script style)

### Losses

- Cross-entropy in all branches
- Metric-learning regularization in experiment branch:
  - hard triplet-style (`old`)
  - multi-similarity (`new`)
- metric losses usually applied only on occupied squares

### Training mechanics

- Adam optimizer, LR around `1e-4` in experiment scripts
- `ReduceLROnPlateau`
- early stopping
- weighted sampling, especially `50_50` empty vs occupied mass balancing

---

## 8) Evaluation Strategy

Primary in experiment branch:

- Macro F1 (`sklearn.metrics.f1_score`, `average='macro'`)
- Fold-wise best checkpointing
- 4-model ensemble on game5:
  - probability fusion to unified 13-class
  - average probabilities across fold models

Additional baseline metrics:

- Piece Accuracy (% correctly predicted squares)
- Board Accuracy (% fully perfect 64-square boards)

OOD/uncertainty support:

- Post-hoc confidence thresholding in `inference_using_kfold.py`
- low-confidence image export for manual inspection

---

## 9) Known Inconsistencies / Risks (Important)

The repository currently contains multiple partially diverged experimental branches and some script-level issues:

1. `experiment2.py` has a combination list construction bug (`todo = combs.append(...)`).
2. `temp.py` contains inconsistent or undefined variables in logging/loss accounting sections.
3. `requirements.txt` likely typo (`dandas`).
4. Different reports/context files claim different headline metrics (e.g., ~84% F1 vs ~94.74% accuracy) and not all are traceably tied to the same script + output artifacts.
5. Some docs reference outputs that may not currently be present in tracked files.

Implication: a final submission should standardize on one definitive experiment pipeline (likely `experiment114.py` or the chosen final branch), regenerate final tables, and align all report claims with reproducible artifacts.

---

## 10) Practical Reproducibility Snapshot

### Typical run commands (as documented)

- Main focused experiment:
  - `python experiment114.py`
- Earlier grid:
  - `python SUPER_SEARCH_9001.py`
- Parallel cluster array:
  - `sbatch sbatch_parallel.sh`
- OOD inference:
  - `python inference_using_kfold.py --models_path <dir> --heads <1|2|3> --dataset_path assets/new_dataset --threshold 0.5`
- Baseline train/eval:
  - `python train.py ...`
  - `python evaluate.py --model_path <pth> --test_dir <dir>`

### Cluster assumptions

- SLURM-based GPU jobs
- Typical resource requests in scripts:
  - 1x RTX 3090
  - 4 CPU
  - 24G RAM
  - up to 1 day wall time

---

## 11) Suggested Single Source of Truth for Final Report

If you want one consistent final narrative, the most coherent basis from current code appears to be:

1. Use `experiment114.py` as the canonical method/results pipeline.
2. Keep report structure from `final_report.md` + `agent4_context.md`.
3. Fill quantitative tables directly from generated CSV artifacts.
4. Explicitly document ablation axes:
   - head type
   - metric loss mode (`old` vs `new`)
   - sampling strategy (where available)
   - freeze/progressive strategy (if included in final chosen branch)
5. Include OOD qualitative analysis from `low_confidence_images.csv` and `ood_inspection/` if generated.

---

## 12) File Coverage Confirmation

This consolidation incorporated content from:

- Context/report files:
  - `agent1_context.md`
  - `agent2_context.md`
  - `agent3_context.md`
  - `agent4_context.md`
  - `final_report.md`
  - `readme.md`
  - `Instructions.md`
- Submission guidelines:
  - `Projects Final Submission .pdf`
- Python source:
  - `SUPER_SEARCH_9001.py`
  - `experiment2.py`
  - `experiment114.py`
  - `parallel_experiment.py`
  - `temp.py`
  - `inference_using_kfold.py`
  - `train.py`
  - `train_v2.py`
  - `evaluate.py`
  - `evaluate_v2.py`
  - `board_generator.py`
- Shell/config:
  - `master_grid.sh`
  - `sbatch_experiment2.sh`
  - `sbatch_experiment114.sh`
  - `sbatch_parallel.sh`
  - `deploy.sh`
  - `.gitignore`
  - `requirements.txt`
- Skill context:
  - `.cursor/skills/examine-skills/SKILL.md`
  - `~/.cursor/skills/task_full_cycle/SKILL.md` (discovered, not applied since irrelevant)

---

End of consolidated context.
