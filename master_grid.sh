#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=SUPER_SEARCH_9001
#SBATCH --output=master_grid_%J.out
#SBATCH --time=3-00:00:00 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Load the cluster's Anaconda module
module load anaconda

# Activate your specific environment
source activate chess_env 

# Run the grid search with unbuffered output to watch live progress
python -u SUPER_SEARCH_9001.py