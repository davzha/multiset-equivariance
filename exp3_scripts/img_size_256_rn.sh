#!/bin/bash
SEED=$1
exp3_scripts/run_single.sh 256 "256img rnfs $SEED" "--decoder_pool rnfs --seed $SEED"