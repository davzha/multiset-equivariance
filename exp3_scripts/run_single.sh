#!/bin/bash
IMGSIZE=$1
NAME=$2
ARGS=$3
DATA_PATH=~/data/clevr

HYPERPARAMS="--lr 1e-3 --batch_size 128 --epochs 100 --lr_drop_epoch 90 --latent_dim 512 --hidden_dim 512"
DECODER="--decoder_iters 40 --decoder_it_schedule --decoder_grad_clip 10"
DATASET="--clevr_image_input --clevr_path $DATA_PATH --clevr_image_size $IMGSIZE --test_after_training"

set -x
python train_exp3.py $DATASET --name "$NAME" --num_data_workers 16 $HYPERPARAMS $DECODER $ARGS
