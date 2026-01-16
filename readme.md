running the code from the cluster:
python train.py --data_dir ./assets/labeled_data --epochs 1

make sure you follow the course guide and update the deploy script with the cluster hostname youre using

to get cluster hostname for ssh- 
cli : hostname -f

asking to get cluster for an hour- 
srun --pty -p rtx2080 --qos=course --gres=gpu:1 --mem=16G --time=1:00:00 /bin/bash