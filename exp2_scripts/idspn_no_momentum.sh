#!/bin/bash
SEED=$1
exp2_scripts/run_single.sh $SEED idspn  "--decoder_momentum 0"
