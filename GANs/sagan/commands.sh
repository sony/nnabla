#!/bin/bash

# 000: default
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 16 -a 4 -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_000 --max-iter 450000 --save-interval 10000


# 001: dynamic scaling and use relu -> bad
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 64 -a 1 -t float -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_001 --max-iter 500000 --save-interval 10000


# 002: dynamic scaling and use relu and 64 batch -> bad
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 64 -a 1 -t float -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_002 --max-iter 500000 --save-interval 10000

# 003: use relu
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 32 -a 2 -t float -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_003 --max-iter 500000 --save-interval 10000


# 004: use relu and 64 batch
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 64 -a 1 -t float -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_004 --max-iter 500000 --save-interval 10000


# 005: use relu and 64 batch
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 64 -a 1 -t float -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_005 --max-iter 1000000 --save-interval 10000


# 006: use relu and 64 batch w/o real input noise
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 32 -a 2 -t float -T /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan -L /home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt --monitor-path ./result/example_006 --max-iter 1000000 --save-interval 10000


# Generate
## random
python generate.py -c cudnn -d 0 -b 36 --monitor-path ./result/example_003 --model-load-path ./result/example_003/params_999999.h5

## class-conditioned
python generate.py -c cudnn -d 0 -b 36 --monitor-path ./result/example_003 --model-load-path ./result/example_003/params_999999.h5 --class-id 1


## class-conditioned and tiled-all
python generate_all.py -c cudnn -d 0 -b 36 --monitor-path ./result/example_003 --model-load-path ./result/example_003/params_999999.h5


# Morph
python morph.py -c cudnn -d 0 -b 8 --n-morphs 8 -t float \
			 --monitor-path ./result/example_005 \
			 --model-load-path ./result/example_005/params_999999.h5 \
       --from-class-id 947 --to-class-id 153


# Match
python match.py -c cudnn -d 0 -t float \
			 -T /data/datasets/imagenet/train_cache_sngan \
			 -L ./dirname_to_label.txt \
			 --monitor-path ./result/example_005 \
			 --model-load-path ./result/example_005/params_999999.h5 \
			 --nnp-inception-model-load-path ~/git/pretrained-imagenet/NNP/Resnet-50_4_178.nnp \
			 --variable-name AveragePooling \
			 --image-size 224 \
			 --nnp-preprocess \
			 --top-n 15 \
			 --class-id 153


# Evaluate
## Inception Score (w/ Inception-V3)
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 -t float \
			 --monitor-path ./result/example_005 \
			 --model-load-path ./result/example_005/params_999999.h5 \
			 --nnp-inception-model-load-path ~/git/pretrained-imagenet/NNP/Inception-v3.nnp \
			 --image-size 320 \
			 --evaluation-metric IS

## Inception Score (w/ ResNet-50)
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 -t float \
			 --monitor-path ./result/example_005 \
			 --model-load-path ./result/example_005/params_999999.h5 \
			 --nnp-inception-model-load-path ~/git/pretrained-imagenet/NNP/Resnet-50_4_178.nnp \
			 --image-size 224 \
			 --evaluation-metric IS \
			 --preprocess

## Frechet Inception Distance (w/ Inception-V3)
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 -t float \
			 --monitor-path ./result/example_005 \
			 --model-load-path ./result/example_005/params_999999.h5 \
			 --nnp-inception-model-load-path ~/git/pretrained-imagenet/NNP/Inception-v3.nnp \
			 --image-size 320 \
			 --evaluation-metric FID --variable-name AveragePooling_2 \
			 -V /data/datasets/imagenet/val_data \
			 -L /data/datasets/imagenet/dirname_to_label.txt

## Frechet Inception Distance (w/ ResNet-50)
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 -t float \
			 --monitor-path ./result/example_005 \
			 --model-load-path ./result/example_005/params_999999.h5 \
			 --nnp-inception-model-load-path ~/git/pretrained-imagenet/NNP/Resnet-50_4_178.nnp \
			 --image-size 224 \
			 --evaluation-metric FID --variable-name AveragePooling \
			 -V /data/datasets/imagenet/val_data \
			 -L /data/datasets/imagenet/dirname_to_label.txt \
			 --nnp-preprocess



