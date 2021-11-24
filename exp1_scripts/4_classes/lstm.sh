#!/bin/bash
SEED=$1
exp1_scripts/run_single.sh lstm hungarian_ce 64 4 $SEED
