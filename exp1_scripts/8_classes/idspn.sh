#!/bin/bash
SEED=$1
exp1_scripts/run_single.sh idspn hungarian_l2 512 8 $SEED "--decoder_grad_clip 40"
