#!/bin/bash
for i in `seq 0 7`
do
    exp2_scripts/idspn_with_momentum.sh $i
    exp2_scripts/idspn_no_momentum.sh $i
    exp2_scripts/dspn.sh $i

    # exp2_scripts/idspn_with_momentum_mean.sh $i
    # exp2_scripts/idspn_with_momentum_sum.sh $i
done
