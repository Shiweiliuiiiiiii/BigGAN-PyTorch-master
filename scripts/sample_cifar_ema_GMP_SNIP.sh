#!/bin/bash
#SBATCH --job-name=Biggan_GMP_SNIP
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=18
#SBATCH -o Biggan_GMP_SNIP.out        # 打印输出的文件
source /home/sliu/miniconda3/etc/profile.d/conda.sh
conda activate slak

python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--sparse --density 0.3 --G_growth gradient --D_growth random \
--experiment_name sparse_fix_sema_imbalanced_densityD_0.3000_densityG_0.0500_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 --sample_npz --load_weights copy0
python inception_tf13.py --experiment_name sparse_fix_sema_imbalanced_densityD_0.3000_densityG_0.0500_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema

python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--sparse --density 0.3 --G_growth gradient --D_growth random \
--experiment_name sparse_fix_sema_imbalanced_densityD_0.3000_densityG_0.1000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 --sample_npz --load_weights copy0
python inception_tf13.py --experiment_name sparse_fix_sema_imbalanced_densityD_0.3000_densityG_0.1000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema

python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--sparse --density 0.3 --G_growth gradient --D_growth random \
--experiment_name sparse_fix_sema_imbalanced_densityD_0.3000_densityG_0.2000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 --sample_npz --load_weights copy0
python inception_tf13.py --experiment_name sparse_fix_sema_imbalanced_densityD_0.3000_densityG_0.2000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema


python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--sparse --density 0.3 --G_growth gradient --D_growth random \
--experiment_name sparse_GMP_sema_imbalanced_densityD_1.0000_densityG_0.0500_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 --sample_npz --load_weights copy0
python inception_tf13.py --experiment_name sparse_GMP_sema_imbalanced_densityD_1.0000_densityG_0.0500_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema


python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--sparse --density 0.3 --G_growth gradient --D_growth random \
--experiment_name sparse_GMP_sema_imbalanced_densityD_1.0000_densityG_0.1000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 --sample_npz --load_weights copy0
python inception_tf13.py --experiment_name sparse_GMP_sema_imbalanced_densityD_1.0000_densityG_0.1000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema

python sample.py \
--shuffle --batch_size 50 --G_batch_size 256 --parallel \
--sparse --density 0.3 --G_growth gradient --D_growth random \
--experiment_name sparse_GMP_sema_imbalanced_densityD_1.0000_densityG_0.2000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0 --sample_npz --load_weights copy0
python inception_tf13.py --experiment_name sparse_GMP_sema_imbalanced_densityD_1.0000_densityG_0.2000_dy_G_D_growth_random_G_growth_gradient_fre_5000_BigGAN_C10_seed1_Gch64_Dch64_bs50_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema
