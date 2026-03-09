#!/bin/bash
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --job-name=experiment114
#SBATCH --output=experiment114_%J.out
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Load the cluster's Anaconda module
module load anaconda [cite: 79]

# Activate your specific environment
# Ensure the environment is deactivated on the manager node before sbatch [cite: 52]
source activate chess_env [cite: 80]

# Run the grid search with unbuffered output
# -u is used to watch live progress and prevent buffered printing [cite: 801, 805]
python -u experiment114.py