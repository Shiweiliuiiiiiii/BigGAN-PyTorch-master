#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J Biggan_PF_global           # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 12-00:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o Biggan_PF_global      # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate torch151
module load cuda/10.0
for pruning_rate in 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
do
  python train_prune_finetune.py \
  --shuffle --batch_size 50 --parallel \
  --sparse --sema --sparse_init pruning --pruning_mode global_G --pruning_rate $pruning_rate --resume  \
  --experiment_name unbalanced_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
  --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
  --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
  --dataset C10 \
  --G_ortho 0.0 \
  --G_attn 0 --D_attn 0 \
  --G_init N02 --D_init N02 \
  --ema --use_ema --ema_start 1000 \
  --test_every 5000 --save_every 2000 --num_best_copies 1 --num_save_copies 1 --seed 0 --load_weights best2
done
for pruning_rate in 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05
do
  python train_prune_finetune.py \
  --shuffle --sema --batch_size 50 --parallel \
  --sparse --sparse_init pruning --pruning_mode global_GD --pruning_rate $pruning_rate --resume  \
  --experiment_name unbalanced_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
  --num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
  --num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
  --dataset C10 \
  --G_ortho 0.0 \
  --G_attn 0 --D_attn 0 \
  --G_init N02 --D_init N02 \
  --ema --use_ema --ema_start 1000 \
  --test_every 250 --save_every 250 --num_best_copies 1 --num_save_copies 1 --seed 0 --load_weights best2
done

conda deactivate torch151