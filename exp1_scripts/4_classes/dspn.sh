#!/bin/bash
SEED=$1
exp1_scripts/run_single.sh dspn hungarian_l2 256 4 $SEED "--decoder_momentum 0"
