#!/bin/bash
SEED=$1
exp3_scripts/run_single.sh 128 "128img fs $SEED" "--decoder_pool fs --seed $SEED"