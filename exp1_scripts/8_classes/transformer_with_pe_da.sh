#!/bin/bash
SEED=$1
exp1_scripts/run_single.sh transformer_with_pe hungarian_ce 128 8 $SEED "--rand_perm"
