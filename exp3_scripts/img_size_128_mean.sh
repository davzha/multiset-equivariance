#!/bin/bash
SEED=$1
exp3_scripts/run_single.sh 128 "128img mean $SEED" "--decoder_pool mean --seed $SEED"