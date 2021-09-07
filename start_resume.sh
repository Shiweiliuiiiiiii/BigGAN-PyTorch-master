#!/bin/bash
densityD=0.5
for densityG in 0.05 0.1 0.2 0.3 0.4
do
  python3.6 train_compression.py \
  --dataset I128_hdf5 --parallel --shuffle  --num_workers 8 --batch_size 256 --load_in_mem  \
  --sparse --sema --resume --sparse_init resume --dy_mode GD --densityG $densityG --densityD $densityD --G_growth gradient --D_growth random --update_frequency 2000 \
  --num_G_accumulations 8 --num_D_accumulations 8 \
  --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B1 0.0 --D_B2 0.999 --G_B2 0.999 \
  --G_attn 64 --D_attn 64 \
  --G_nl inplace_relu --D_nl inplace_relu \
  --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
  --G_ortho 0.0 \
  --G_init ortho --D_init ortho \
  --hier --dim_z 120 --shared_dim 128 --G_shared \
  --G_eval_mode \
  --G_ch 48 --D_ch 96 \
  --ema --use_ema --ema_start 20000 \
  --test_every 500 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
  --use_multiepoch_sampler \
  --model BigGAN \
  --pretrain_path /apdcephfs/share_1367250/yuesongtian/pretrained_models/138k/D.pth \
  --G_pretrain_path /apdcephfs/share_1367250/yuesongtian/BigGAN_results/weights/hinge_baseline_VanillaCls_orthoInit/G_ema_best4.pth \
  --data_root /apdcephfs/share_1367250/yuesongtian/ \
  --experiment_name hinge_G1-2_D1-1_dgl_finetune \
  --g_loss loss_hinge_genDual --d_loss loss_hinge_dis \
  --resume \
  --num_epochs 1000 \
  --load_weights best4 \
done


