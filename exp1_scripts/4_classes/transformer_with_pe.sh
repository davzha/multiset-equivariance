#!/bin/bash
SEED=$1
exp1_scripts/run_single.sh transformer_with_pe hungarian_ce 64 4 $SEED
