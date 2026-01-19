# ♟️ Chess Deep Learning Project - BGU Cluster Workflow

This guide documents the setup, data management, and training workflow for the Chess Deep Learning project on the Ben-Gurion University (BGU) SLURM cluster.

## 📋 Prerequisites

Before starting, ensure you have the following on your local machine:
1.  **VSCode** (Visual Studio Code).
2.  **Git** (Recommended: Install with Git Bash for Windows).
3.  **MobaXterm** or **WinSCP** (For file transfer via SFTP).
4.  **BGU VPN**: Required if connecting from outside the university network.

---

## 🚀 1. Cluster Setup & Environment

### Connect via SSH
Open your terminal and connect to the cluster:
```bash
ssh <your_username>@slurm.bgu.ac.il

## Create the Conda Environment:

# 1. Load Anaconda module
module load anaconda

# 2. Create a new environment named 'chess_env'
conda create --name chess_env python=3.9 -y

# 3. Activate the environment
source activate chess_env

# 4. Install dependencies
pip install torch torchvision pandas numpy Pillow tqdm
```


---

## 🚀 Data management
```bash
# Navigate to the assets folder
cd ~/chess-DL-project/assets

# Create the target directory and move the zip there
mkdir -p labeled_data
mv *.zip labeled_data/
cd labeled_data

# Extract all sub-zips into their own folders
for f in *.zip; do
  dirname="${f%.zip}"      # Get filename without extension
  unzip -o "$f" -d "$dirname"  # Unzip into a folder with that name
done

# Cleanup: Remove the zip files to save space
rm *.zip
```
### Runnning the training - finally!
srun --pty -p rtx2080 --qos=course --gres=gpu:1 --mem=16G --time=1:00:00 /bin/bash
module load anaconda
source activate chess_env
python train.py --data_dir ./assets/labeled_data --epochs 1 --batch_size 4

## upload to git-
sh deploy.sh (from git bash!)
