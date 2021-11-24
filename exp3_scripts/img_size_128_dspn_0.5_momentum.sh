#!/bin/bash
SEED=$1
exp3_scripts/run_single.sh 128 "128img dspn 0.5mom fs $SEED" "--decoder_pool fs --model dspn --decoder_momentum 0.5 --seed $SEED"