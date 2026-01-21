ssh nirhoro@slurm.bgu.ac.il

sinteractive --qos course --part gtx1080 --gpu 1 --time 0-08:00:00

see https://moodle.bgu.ac.il/moodle/pluginfile.php/5077646/mod_resource/content/4/Accessing%20an%20Interactive%20GPU%20on%20the%20BGU%20Cluster.pdf

now ssh with what it gave you

module load anaconda

only once in your lifetime:
conda create --name chess_env python=3.9 -y

source activate chess_env

only once in your life time

pip install -r requirements.txt

will copy to where ever it was written the things in nir's cluster
scp -r nirhoro@slurm.bgu.ac.il:/home/nirhoro/chess-DL-project

python train.py --data_dir ./assets/labeled_data --epochs 1 --batch_size 32
