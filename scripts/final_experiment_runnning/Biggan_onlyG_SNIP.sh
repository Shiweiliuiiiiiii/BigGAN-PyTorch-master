#!/bin/bash
#SBATCH --job-name=Biggan_onlyG_GMP
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o Biggan_onlyG_GMP.out        # 打印输出的文件
source /home/sliu/miniconda3/etc/profile.d/conda.sh
conda activate slak
4
for densityG in 0.05 0.1 0.2
do
  python train.py \
  --shuffle --batch_size 50 --parallel \
  --sparse --sema --dy_mode G --imbalanced --densityD 1.0 --densityG $densityG --G_growth gradient --D_growth random --update_frequency 5000 \
  --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500  --sparse_mode GMP --sparse_init dense --initial_prune_time 0.0 \
  --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
  --dataset C10 \
  --G_ortho 0.0 \
  --G_attn 0 --D_attn 0 \
  --G_init N02 --D_init N02 \
  --ema --use_ema --ema_start 1000 \
  --test_every 5000 --save_every 2000 --num_best_copies 1 --num_save_copies 1 --seed 1
done