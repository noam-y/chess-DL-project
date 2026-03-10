#!/bin/bash
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --job-name=fine_tune_experiment
#SBATCH --output=fine_tune_results_%J.out
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Load the cluster's Anaconda module
module load anaconda

# Activate your specific environment
# Ensure the environment is deactivated on the manager node before sbatch
source activate chess_env

# Run fine-tuning with unbuffered output
# -u is used to watch live progress and prevent buffered printing
python -u fine_tune.py
