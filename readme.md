# ♟️ Chess Deep Learning Project - BGU Cluster Workflow

This guide documents the setup, data management, and training workflow for the Chess Deep Learning project on the Ben-Gurion University (BGU) SLURM cluster.

## 📋 Prerequisites

Before starting, ensure you have the following on your local machine:
1.  **Git** 
2.  **BGU VPN**

---

## 🚀 Final Pipeline (Train + Evaluate)

Run the following commands in order:

Connect to the BGU cluster. 🔐
```bash
ssh <your_username>@slurm.bgu.ac.il
```

Clone the repository. 📦
```bash
git clone https://github.com/noam-y/chess-DL-project
cd chess-DL-project
```

Install dependencies. 🧩
```bash
pip install -r requirements.txt
```

Download and extract the dataset. 📥
```bash
python setup_dataset.py
```

Preprocess the dataset. 🛠️
```bash
python preprocess_dataset.py
```

Request a GPU session on the cluster. 🚀
```bash
srun --partition=course --qos=course --gres=gpu:rtx_3090:1 --time=04:00:00 --pty bash -i
```

Train the model. 🧠
```bash
python train.py
```

Evaluate on unseen data. 📊
```bash
python evaluate.py
```

### Troubleshooting: can't connect after receiving a cluster node?

Check your queue and node assignment. 🔎

```bash
squeue --me
```

Then copy the node name from `NODELIST` and connect to it via SSH.
