#!/bin/bash
SEED=$1
exp3_scripts/run_single.sh 128 "128img idspn 0.5mom fs $SEED" "--seed $SEED --decoder_pool fs --decoder_momentum 0.5"