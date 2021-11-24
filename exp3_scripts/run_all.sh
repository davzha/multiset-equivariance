#!/bin/bash
for seed in {0..7}
do
    exp3_scripts/img_size_128.sh $seed
    exp3_scripts/img_size_256.sh $seed
done

# optional
#for seed in {0..7}
#do
#    # relation nets
#    exp3_scripts/img_size_128_rn.sh $seed
#    exp3_scripts/img_size_256_rn.sh $seed
#
#    # idspn + sum and mean baselines
#    exp3_scripts/img_size_128_sum.sh $seed
#    exp3_scripts/img_size_128_mean.sh $seed
#
#    # dspn vs idspn like-for-like at lower momentum
#    exp3_scripts/img_size_128_dspn_0.5_momentum.sh $seed
#    exp3_scripts/img_size_128_idspn_0.5_momentum.sh $seed
#done
