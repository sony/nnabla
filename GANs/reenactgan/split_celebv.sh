#!/bin/bash

DATAROOT=$1
NUM_TRAIN=$2

if [ -z "$DATAROOT" ]; then
    echo "You must specify the directory containing CelebV data."
    exit 1
fi

if [ -z "$NUM_TRAIN" ]; then
    echo "number of the training data not given: 30000 used."
    NUM_TRAIN=30000
fi

for PERSON in Donald_Trump  Emmanuel_Macron  Jack_Ma  Kathleen  Theresa_May; do
    split $DATAROOT/$PERSON/all_98pt.txt tmp_split_ -l $NUM_TRAIN
    mv tmp_split_aa $DATAROOT/$PERSON/train_98pt.txt
    mv tmp_split_ab $DATAROOT/$PERSON/test_98pt.txt
    echo "create train_98pt.txt and test_98pt.txt for $PERSON at $DATAROOT/$PERSON"
done
