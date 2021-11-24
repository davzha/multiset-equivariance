#!/bin/bash
SEED=$1
exp1_scripts/run_single.sh idspn hungarian_l2 256 4 $SEED "--decoder_grad_clip 40"
