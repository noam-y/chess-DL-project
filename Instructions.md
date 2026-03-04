# ChessNet Grid Search

This repository contains the massive 36-combination grid search pipeline to find the ultimate chess piece classification architecture.

## How to run the Master Grid Search on the GPU Cluster:

1. SSH into the cluster.
2. Ensure you are in this repository's directory.
3. Make sure the dataset is properly located at `assets/new_dataset/`.
4. Deactivate any active environments and submit the job:
   ```bash
   conda deactivate
   sbatch master_grid.sh