#!/bin/bash
SEED=$1
MODEL=$2
EXTRA_ARGS=$3

COMMON="--epochs 40 --lr 1e-3 --batch_size 128 --project random_sets --decoder_learn_init_set"

set -x
for size in 2 4 8 16 32
do
    for dim in 2 4 8 16 32
    do
        for iters in 10 20 40
        do
            RUN_NAME="${size}x${dim} $MODEL it${iters} $SEED"
            python train_exp2.py --name "$RUN_NAME" $COMMON $EXTRA_ARGS --seed $SEED --model $MODEL --set_size $size --set_dim $dim --decoder_iters $iters
        done
    done
done
