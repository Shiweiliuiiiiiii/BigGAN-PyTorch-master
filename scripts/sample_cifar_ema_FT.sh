#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J GGG              # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10      # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1           # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p p40                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 0-05:00:00            # 运行的最长时间 day-hour:minute:second，但是请按需设置，不要浪费过多时间，否则影响系统效率
#SBATCH -o FID_FT_only_G.out        # 打印输出的文件
source /public/data2/software/software/anaconda3/bin/activate
conda activate torch151
module load cuda/10.0
#for density in  0.2 0.1 0.05
#do
python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--sparse --density 0.03 --dy_mode GD --G_growth gradient --D_growth random  \
--experiment_name unbalanced_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema_FT_global_G_density0.000100 \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 --sample_npz --load_weights best0
#
conda deactivate
conda activate GAN1
python inception_tf13.py --experiment_name unbalanced_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema_FT_global_G_density0.000100
conda deactivate

#conda activate GAN1
#module load cuda/10.0
#for p in 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05
#do
#  python inception_tf13_prune.py --pruning_rate $p --experiment_name unbalanced_BigGAN_C10_seed0_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema
#done
#conda deactivate GAN1


