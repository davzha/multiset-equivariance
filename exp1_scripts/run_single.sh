#!/bin/bash
MODEL=$1
LOSS=$2
DIM=$3
N_CLASSES=$4
SEED=$5
EXTRA_ARGS=$6

args="--num_ray_workers 6 --set_size 64 --set_dim 64 --input_dim $N_CLASSES --latent_dim $DIM --hidden_dim $DIM --loss $LOSS --seed $SEED --model $MODEL $EXTRA_ARGS"

set -x
python train_exp1.py $args --epochs 10000 --check_val_every_n_epoch 100 --dataset_size 640 --name "$N_CLASSES 640 $MODEL 64x$N_CLASSES $SEED"
python train_exp1.py $args --epochs 1000 --check_val_every_n_epoch 10 --dataset_size 6400 --name "$N_CLASSES 6400 $MODEL 64x$N_CLASSES $SEED"
python train_exp1.py $args --epochs 100 --check_val_every_n_epoch 1 --dataset_size 64000 --name "$N_CLASSES 64000 $MODEL 64x$N_CLASSES $SEED"
