# Inception Score and FID (Fréchet Inception Distance)

This is the implementation of Inception Score and FID by NNabla.
These scores are often used to evaluate how *good/real* the generated images are.

## Requirements

To get the same score as original implementations, you need to prepare the pretrained Inception v3 weights. Although nnabla provides Inception v3 as a pretrained model, the Inception v3 used to calculate these scores are slightly different in terms of internal architecture and configurations. We have converted the original weights provided by TensorFlow and when you execute either `inception_score.py` or `fid.py`, it will automatically download the weights and store it in this directory. The corresponding Inception v3 architecture can be found in `inceptionv3.py`.

## Calculate Inception Score and FID

### Prerequisite

It is better to add `metric`'s path to the `PYTHONPATH` for convenience.

```
export PYTHONPATH=/path/to/the/nnabla-examples/utils/neu:$PYTHONPATH
```

### Simple Usage for Computing Inception Score

To calculate inception score on some fake images set, you need to specify a directory which contains the fake images, or, you can specify a text file in which fake images path are listed.
You can specify the path to the pretrained weights, but by default this script tries to use the weights which will be automatically downloaded if not present, so you can omit that.

```
python -m metrics.gan_eval.inception_score <path to the directory or text file> 
```

Then you get the results like;

```
2020-04-15 04:46:13,219 [nnabla][INFO]: Initializing CPU extension...
2020-04-15 04:46:16,921 [nnabla][INFO]: Initializing CUDA extension...
2020-04-15 04:46:16,923 [nnabla][INFO]: Initializing cuDNN extension...
calculating all features of fake data...
loading images...
Finished extracting features. Calculating inception Score...
Image Sets: <image set used for calculation>
batch size: 16
split size: 1
Inception Score: 38.896
std: 0.000
```

In the original implementation, Inception Score is obtained by averaging the scores calculated on split subsets of given images (with standard deviation). In this script, the score is calculated using whole images (not averaged on multiple scores) by default. You can compute the score in the same manner as the original by adding `--split` option. They set the split size 10 (so specify `--split 10`).


Needless to say, you can execute the script in this directory. Also, you can alternatively use your own weights for Inception v3 with `--params-path` option.
```
python inception_score.py <path to the directory or text file> --params-path <path to the pretrained weights>
```

### Simple Usage for Computing Fréchet Inception Distance

To calculate Fréchet Inception Distance, you need to have **both of real and fake images data** and specify them when you run this script. For each input data, you need to specify it as one of the following;
* a directory which contains the fake images.
* a text file in which fake images path are listed.
* `.npz` file which contains 2 data fields, one is `mu` and another is `sigma`.

Third option (using `.npz`) is useful when you have pre-calculated statistics of real/fake images. Particularly, the original authors kindly [provide the pre-calculated statistics](http://bioinf.jku.at/research/ttur/) of various dataset as `.npz` format, so you can just use that. Also you can save the computed statistics with `--save-stats` option. For more details, please see the script.
You can specify the path to the pretrained weights here as well, especially if you want to use your own weights.

```
python -m metrics.gan_eval.fid <path to the directory, text file, or .npz file of REAL data> \
                               <path to the directory, text file, or .npz file of FAKE data> \
                               --params-path <path to the pretrained weights, can be omitted>
```

Then you get the results like;

```
2020-04-10 10:08:47,272 [nnabla][INFO]: Initializing CPU extension...
2020-04-10 10:08:47,518 [nnabla][INFO]: Initializing CUDA extension...
2020-04-10 10:08:47,519 [nnabla][INFO]: Initializing cuDNN extension...
Computing statistics...
loading images...
100%|##################################################################################################################| 10000/10000 [07:36<00:00, 22.07it/s]
Image Set 1: <image set 1 used for calculation>
Image Set 2: <image set 2 used for calculation>
batch size: 16
Frechet Inception Distance: 55.671
```


## Note

### About the number of samples
The number of samples to calculate the Gaussian statistics (mean and covariance) **should be greater than the dimension of the coding layer, here 2048** in this case. Otherwise the covariance is not full rank resulting in complex numbers and nans by calculating the square root. The original author mentions they recommend using **a minimum sample size of 10,000** to calculate the FID, otherwise the true FID of the generator is underestimated.

### About the Inception v3 model
As mentioned before, the architecture of Inception v3 is slightly different from the one as we know it. For example;

* the final output feature has 1008 dimensions, not 1000.
* some discrepancy in module design.

Since these scores completely depend on the image features extracted by **this** Inception v3, which is trained on ImageNet classification task, sometimes the obtained scores might not be good criteria for evaluation. For example, using **this** Inception Score or FID implementation to evaluate the quality of generated human's face might not be a good idea because of a kind of **domain gap**. One possible solution would be replacing the feature extractor with appropriate one.

This is easily done by modifying the code (`fid.py` in this case) a bit like;

```
def get_features(input_images):
    # feature = construct_inceptionv3(input_images)
    feature = your_network(input_images)
    return feature
```

Here, `your_network` must have the similar I/O to the original Inception v3, in other words, it must accept images as input and return flattened image features (such as output of a pooling layer).


### About the resizing method
Since Inception v3 takes fixed size input whose shape is (B, 3, 299, 299). Images used to calculate these scores need to be resized in advance. In the original implementation, TensorFlow's bilinear interpolation is applied for resizing. Since their resizing method returns a slightly different result compared to those by other libraries, we use almost the same resizing implementation as TensorFlow's bilinear interpolation.

In addition, we use PIL == 6.2.1 and imageio == 2.6.1. Other versions might cause an error. You can replace `imread` function with another, but the result may be slightly different due to the different encoding algorithm.


### Validation

This is the comparison table of the Inception Score and FID obtained by the reference (TensorFlow) implementation and NNabla's. Both scores are calculated using the images generated by [NNabla's SAGAN](https://github.com/sony/nnabla-examples/tree/master/GANs/sagan) and ImageNet validation dataset's statistics provided [here](http://bioinf.jku.at/research/ttur/) for FID. Each set has 10000 generated images. Note that scores here are different from those reported in our SAGAN's repository, but this is due to the different pretrained weights used.
Also, note that, although we confirm that we are able to get almost the same score as the original implementation as shown below, due to the non-deterministic algorithms used in cuDNN, results *could* become slightly different.

### Inception Score

Each pair stands for (score, std). In this comparison, split size is 10 and batch size is 16.

|  | Set 1 | Set 2 | Set 3 | Set 4 | Set 5 | Set 6 | Set 7 | Set 8 | Set 9 | Set 10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TF     | (19.898, 2.479) | (19.811, 2.358) | (19.680, 2.693) | (19.887, 2.082) | (19.728, 2.323) | (19.971, 2.432) | (19.829, 2.498) | (19.646, 2.336) | (20.143, 2.370) | (19.889, 2.713) |
| NNabla | (19.898, 2.479) | (19.811, 2.358) | (19.680, 2.692) | (19.887, 2.082) | (19.728, 2.323) | (19.971, 2.432) | (19.829, 2.499) | (19.646, 2.336) | (20.143, 2.370) | (19.889, 2.713) |

### FID

batch size is 50.

|  | Set 1 | Set 2 | Set 3 | Set 4 | Set 5 | Set 6 | Set 7 | Set 8 | Set 9 | Set 10 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TF     | 55.32459 | 55.75309 | 55.70617 | 55.86620 | 55.49323 | 55.60468 | 55.55891 | 55.47711 | 55.24311 | 55.67127 |
| NNabla | 55.32460 | 55.75308 | 55.70619 | 55.86624 | 55.49322 | 55.60468 | 55.55888 | 55.47706 | 55.24310 | 55.67147 |


## Reference

### Inception Score

* Paper: [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
* Github: [openai/improved-gan](https://github.com/openai/improved-gan)

### FID

* Paper: [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf)
* Github: [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)

### Others
* Original InceptionV3 pretrained models: [Download Link](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)
* [TensorFlow's ResizeBilinear implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/resize_bilinear_op.cc)
