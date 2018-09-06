#!/bin/bash

python classification.py -d 0 -c "cudnn" --net cifar10_shift_csc_prediction  --maps 64 --monitor-path csc-maps_64 --model-save-path csc-maps_64 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_csc_prediction  --maps 128 --monitor-path csc-maps_128 --model-save-path csc-maps_128 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_csc_prediction  --maps 192 --monitor-path csc-maps_192 --model-save-path csc-maps_192 --type-config half

python classification.py -d 0 -c "cudnn" --net cifar10_shift_csc_prediction  --maps 64 -p 0.5 --monitor-path csc-maps_64-p_5 --model-save-path csc-maps_64-p_5 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_csc_prediction  --maps 128 -p 0.5 --monitor-path csc-maps_128-p_5 --model-save-path csc-maps_128-p_5 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_csc_prediction  --maps 192 -p 0.5 --monitor-path csc-maps_192-p_5 --model-save-path csc-maps_192-p_5

python classification.py -d 0 -c "cudnn" --net cifar10_shift_sc2_prediction  --maps 64 --monitor-path sc2-maps_64 --model-save-path sc2-maps_64 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_sc2_prediction  --maps 128 --monitor-path sc2-maps_128 --model-save-path sc2-maps_128 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_sc2_prediction  --maps 192 --monitor-path sc2-maps_192 --model-save-path sc2-maps_192 --type-config half

python classification.py -d 0 -c "cudnn" --net cifar10_shift_sc2_prediction  --maps 64 -p 0.5 --monitor-path sc2-maps_64-p_5 --model-save-path sc2-maps_64-p_5 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_sc2_prediction  --maps 128 -p 0.5 --monitor-path sc2-maps_128-p_5 --model-save-path sc2-maps_128-p_5 --type-config half
python classification.py -d 0 -c "cudnn" --net cifar10_shift_sc2_prediction  --maps 192 -p 0.5 --monitor-path sc2-maps_192-p_5 --model-save-path sc2-maps_192-p_5 --type-config half

