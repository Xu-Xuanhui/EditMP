#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7--multiprocessing-distributed False \
# --d_depth 3 \
# --g_depth 5,4,2 \
os.system(f"  python /home/robot/Documents/EditMP/train_derived_mp.py \
-gen_bs 16 \
-dis_bs 16 \
--steps 8 \
--max_iter 500000 \
--gen_model ViT_custom_mp_v5 \
--dis_model ViT_custom_mp_v5 \
--df_dim 6 \
--d_heads 2 \
--d_depth 5 \
--g_depth 5,5,3 \
--dropout 0 \
--latent_dim 8 \
--gf_dim 6 \
--num_workers 10 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--g_norm bn \
--d_norm bn \
--optimizer adam \
--loss standard \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--n_critic 4 \
--grow_steps 0 0 \
--ema 0.9999 \
--diff_aug None \
--lr_decay \
--train \
--Expert_Trajs_path /home/robot/Documents/EditMP/exps/data/curves_2d.npy \
--exp_name ractangles_train_standardloss")
