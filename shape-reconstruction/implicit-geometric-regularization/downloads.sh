#!/usr/bin/env bash

DPATH=stanford_3d_scanning_datasets
mkdir -p $DPATH

# .tar.gz
datasets=(
    "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
    "http://graphics.stanford.edu/pub/3Dscanrep/drill.tar.gz"
    "http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz"
    "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz"
)
for dataset in "${datasets[@]}"; do
    wget ${dataset} -P ${DPATH}
done

for dataset in $(ls ${DPATH} | grep "tar.gz");do
    tar zxvf ${DPATH}/${dataset} -C ${DPATH}
done

# .gz
datasets=(
    "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_manuscript.ply.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz"
)
for dataset in "${datasets[@]}"; do
    wget ${dataset} -P ${DPATH}
done

for dataset in $(ls ${DPATH} | grep "ply.gz");do
    gzip -d ${DPATH}/${dataset}
done
