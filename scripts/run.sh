#!/bin/bash

# export ROOT_DIR=/data/yunqi/3DVision/uncert-nerf/data/Replica

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --root_dir $ROOT_DIR/office1 --dataset_name replica \
#     --exp_name office1 \
#     --num_epochs 5 --batch_size 16384 --lr 2e-2 --eval_lpips



export ROOT_DIR=/data/yunqi/3DVision/uncert-nerf/data/nerf

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --root_dir $ROOT_DIR/lego --dataset_name nerf \
#     --num_epochs 20 --batch_size 8192 --downsample 0.5\
#     --optimizer adam --lr 5e-4 \
#     --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
#     --exp_name nerf_norm 

CUDA_VISIBLE_DEVICES=2 python train.py \
    --root_dir $ROOT_DIR/lego --dataset_name nerf \
    --num_epochs 20 --batch_size 8192 --downsample 0.5\
    --optimizer adam --lr 5e-4 \
    --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
    --uncertainty_loss \
    --exp_name nerf_uncert