#!/bin/bash
min=12
max=17
PREFIX="exp1_scripts/8_classes"

for i in `seq $min $max`
do
    $PREFIX/idspn.sh $i

    $PREFIX/dspn.sh $i
    $PREFIX/lstm_da.sh $i
    $PREFIX/transformer_rnd_pe_da.sh $i
    $PREFIX/transformer_with_pe_da.sh $i
done