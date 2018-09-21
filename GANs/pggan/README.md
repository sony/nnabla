# Progressive Growing of GANs

## Overview

Reproduction of the work, "Progressive Growing of GANs for Improved Quality, Stability, and Variation" by NNabla. 

### Datasets

For the training, the following dataset(s) need to be available:

- [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - Download `img_align_celeba_png`.
  - Decompress via `7za e img_align_celeba_png.7z.001`.
- ([LSUN](http://www.yf.io/p/lsun) and [LSUN Challenge](http://lsun.cs.princeton.edu/2016/))
- (CelebA-HQ)

### Configuration

In `args.py`, there are configurations for training PGGANs, 
generating images, and validating trained models using a certain metric.


### Training

Train the progressive growing of GANs with the following command,

```python
python train.py --device-id 0 \
                --img-path <path to images> \
                --monitor-path <monitor path>
```

It takes about 1 day using the single Tesla V100. 
After the training finishes, you can find the parameters of the trained model, 
the generated images during the training, the training configuration, 
the log of losses, and etc in the `<monitor path>`.

### Generation

For generating images, run

```python
python generate.py --device-id 0 \
                   --model-load-path <path to model> \
                   --monitor-path <monitor path>
```

The generated images are located the `<monitor path>`.

### Validation

Validate models using some metrics.

```python
python validate.py --device-id 0 \
                   --img-path <path to images> \
                   --evaluation-metric <swd or ms-ssim> \
                   --monitor-path <monitor path>
```

The log of the validation metric is located in the `<monitor path>`.

## NOTE
- Currently, we are using LSGAN.
- [TODO] Some works on LSUN dataset
- [TODO] CelebA-HQ

## References

- Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen, "Progressive Growing of GANs for Improved Quality, Stability, and Variation", arXiv:1710.10196.
- https://github.com/tkarras/progressive_growing_of_gans
- https://github.com/tkarras/progressive_growing_of_gans/tree/original-theano-version

## Acknowledgment

This work was mostly done by the intern.

