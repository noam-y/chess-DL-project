# 1. Abstract

* **The problem:** Accurate classification of chess pieces from standard $96 \times 96$ RGB board square images, addressing severe class imbalance (empty squares vastly outnumber specific pieces) and high intra-class variance.
* **Your approach:** A configurable, multi-head Convolutional Neural Network utilizing a pre-trained ResNet-18 backbone. The architecture branches into 1, 2, or 3 classification heads to isolate hierarchical features (occupancy, color, piece type). Training integrates progressive unfreezing, customized weighted random sampling (50/50 split between empty and occupied), and metric learning (Triplet Loss and Multi-Similarity Loss) on the feature embeddings.
* **Main results:** A 72-permutation grid search utilizing 4-fold cross-validation identifies the optimal configuration. The final evaluation is conducted via a 4-model ensemble on a strictly unseen test set (game5), measured by Macro F1-score.

# 2. Introduction

* **Task description and motivation:** Digitizing physical or digital chess boards requires robust piece classification at the square level.
* **Challenges and goals:** * Class Imbalance: Empty squares ('e') dominate the dataset.
* Visual Similarity: Distinguishing between pawns and bishops, or white vs. black pieces under varying lighting/rendering conditions.
* Goal: Maximize classification accuracy across all 13 states (12 pieces + 1 empty) utilizing an optimized, ensembled CNN architecture.


* **Main contributions:**
* Development of a dynamic multi-head ResNet-18 framework decoupling occupancy, color, and piece classification.
* Implementation of progressive backbone unfreezing to preserve pre-trained weights while adapting to the target domain.
* Integration and comparative analysis of Triplet Loss vs. Multi-Similarity Loss for clustered feature space representation.



# 3. Related Work

* **Relevant architectures:** Pre-trained CNNs for image classification (ResNet-18). Alternative architectures considered include VGG11 and MobileNetV2.
* **Metric Learning:** Utilization of Triplet Loss for hard-negative mining and Multi-Similarity Loss for generalized pair weighting in feature embedding spaces.
* **Differences from prior work:** Replaces flat 13-class prediction with hierarchical prediction heads (Occupancy $\rightarrow$ Color $\rightarrow$ Piece) combined with metric learning on the intermediate embeddings.

# 4. Method

* **Model architecture:** `ConfigurableChessResNet`.
* Backbone: ResNet-18 (ImageNet weights) with the final fully connected layer removed.
* Head Configurations:
* 1-Head: Single linear layer mapping to 13 classes.
* 2-Head: Occupancy (2 classes), Piece Type (12 classes).
* 3-Head: Occupancy (2 classes), Color (2 classes), Piece Type (6 classes).




* **Input/output representation:**
* Input: $96 \times 96$ RGB image tensors.
* Output: Logits corresponding to the active head configuration, subsequently mapped to the 13 unified classes via deterministic logic (e.g., if occupancy is 0, output is 'e').


* **Training procedure:**
* Optimizer: Adam (Base LR = 0.0001, Weight Decay = 1e-4).
* Scheduler: ReduceLROnPlateau (factor=0.1, patience=2).
* Early Stopping: Patience 40 epochs. Maximum 100 epochs.
* Progressive Unfreezing: Backbone layers (layer4 to conv1) sequentially unfreezing at 10-epoch intervals. Batch normalization is frozen for locked layers.


* **Loss functions:** * Standard: CrossEntropyLoss for all classification heads.
* Metric Learning: Triplet Loss or Multi-Similarity Loss applied to the flattened backbone features exclusively for occupied squares.


* **Preprocessing/postprocessing:**
* Preprocessing: Resize to $96 \times 96$, ToTensor, ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
* Sampling: `WeightedRandomSampler` enforcing a 50/50 probability distribution between empty squares and all combined occupied squares.



# 5. Experiments

* **Dataset description and splits:** * Data source: Custom `assets/new_dataset` directory containing tagged FEN images from chess games.
* Validation Split: 4-Fold Cross-Validation utilizing game2, game4, game6, game7.
* Test Split: game5 is strictly isolated for final ensemble evaluation.


* **Evaluation metrics:** Macro F1-score (zero_division=0) to account for class imbalances.
* 
**Hardware Execution:** Executed on the BGU ISE-CS-DT GPU cluster utilizing RTX 3090 GPUs. Jobs allocated 1 GPU, 4 CPUs, and 24G RAM.


* **Baselines and comparisons:** * Baseline: 1-Head, no sampling, old Triplet Loss, static freezing.
* Comparisons: Grid search evaluating 72 distinct hyperparameter combinations.


* **Quantitative results:** Final output generated via aggregated CSV files comparing Mean 4-Fold F1 and Ensemble Test F1 (game5).

# 6. Ablation Study

The 72-permutation grid search inherently serves as a comprehensive ablation study, isolating the following variables:

* **Head Configuration (1 vs. 2 vs. 3):** Determines if isolating the classification of occupancy and color improves feature extraction compared to a flat 13-class prediction.
* **Loss Topology (Old Triplet vs. New Multi-Similarity):** Evaluates the impact of advanced metric learning on feature embedding clustering.
* **Freezing Strategy (Static vs. Progressive):** Tests if progressive unfreezing yields superior domain adaptation compared to immediate global gradient calculation or strict freezing.
* **Sampling Strategy (None vs. 50_50):** Quantifies the degradation in F1-score when class imbalance is unmitigated by the `WeightedRandomSampler`.
* **Batch Size (16 vs. 32):** Assesses the impact of batch size on gradient stability and metric loss efficacy.

# 7. What Did Not Work

* 
**Parallel Execution via Slurm Job Arrays:** Attempted to execute 14 grid search tasks concurrently utilizing `#SBATCH --array=1-14`. Execution reverted to sequential processing due to strict `QOSMaxGRESPerUser` limits restricting the account to 1 concurrent GPU.


* **Shared I/O Output in Parallelism:** Initial design mapped all array outputs to a singular `running_results_exp_114.csv` file, which causes race conditions and data overwriting in parallel environments. Resolved by exporting isolated `results_task_X.csv` files per node.
* **Environment Variable Parsing:** Code implementation failed using `sys.getenv('SLURM_ARRAY_TASK_ID')` due to an invalid module attribute call. Code was refactored to use the correct `os.getenv` method.

# 8. Discussion / Limitations

* **Limitations of the method:** * Ensemble inference requires processing the input through four distinct models, increasing computational overhead during deployment.
* The `QOSMaxGRESPerUser` cluster limitation drastically bottlenecks experimental iteration speed.




* **Possible future improvements:** * Substituting the ResNet-18 backbone with MobileNetV2 for low-latency, parameter-efficient inference.
* Implementing a data aggregation script to automatically merge the isolated `results_task_X.csv` files generated by the Slurm array.