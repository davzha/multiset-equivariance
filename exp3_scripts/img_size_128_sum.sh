#!/bin/bash
SEED=$1
exp3_scripts/run_single.sh 128 "128img sum $SEED" "--decoder_pool sum --seed $SEED"