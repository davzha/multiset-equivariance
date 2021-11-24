#!/bin/bash
SEED=$1
exp2_scripts/run_single.sh $SEED dspn "--decoder_momentum 0"
