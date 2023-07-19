#!/bin/bash

export ROOT_DIR=/data/yunqi/3DVision/uncert-nerf/data/Replica

CUDA_VISIBLE_DEVICES=1 python train.py \
    --root_dir $ROOT_DIR/office1 --dataset_name replica \
    --exp_name office1 \
    --num_epochs 5 --batch_size 16384 --lr 2e-2 --eval_lpips
