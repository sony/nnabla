# LPIPS

This is the implementation of LPIPS (Learned Perceptual Image Patch Similarity) metric by NNabla.
These scores can be used to measure how *perceptually close* 2 images are. As the name implies, LPIPS measures similarity based on image features extracted by a trained deep neural network. In the paper, mainly AlexNet, VGG and SqueezeNet are used as feature extractors, and even simple architecture such as AlexNet is reported to work well. So, in this script, AlexNet is used by default.

## Requirements

Since LPIPS is measured based on the image features, you need to have a pretrained network for feature extraction.
The paper's authors kindly provide the pretrained weights for calculating LPIPS, and we already converted them for NNabla. 
When you execute `lpips.py` for the first time, it will automatically download the converted pretrained weights and store them in this directory. If you want to use your own weights, you need to specify the path to a directory containing your weights with `--params-dir` option. 


## Calculate LPIPS

### Prerequisite

It is better to add `metric`'s path to the `PYTHONPATH` for convenience.

```
export PYTHONPATH=/path/to/the/nnabla-examples/utils/neu:$PYTHONPATH
```


### Simple Usage

You can specify an image pair to measure LPIPS between them.

```
python -m metrics.lpips.compute <path to a reference image> <path to another image>
```

It should output the results like below.

```
2020-04-16 10:12:33,739 [nnabla][INFO]: Initializing CPU extension...
2020-04-16 10:12:34,200 [nnabla][INFO]: Initializing CUDA extension...
2020-04-16 10:12:34,201 [nnabla][INFO]: Initializing cuDNN extension...
Use AlexNet's features
LPIPS: 0.722
```

### Another Usage
You can specify 2 directories containing reference images and compared images like;

```
python -m metrics.lpips.compute <path to a directory for reference images> <path to the another directory> \
                                -o <filename which the result is recorded>
```

Note that names of images (stored in both directories) should match each other. The names don't have be exactly the same, but should have some kind of correspondence. Otherwise this script might computes LPIPS between totally unrelated images. This is because there is no guarantee that images are retrieved in the same manner in both directories. Alternatively, you can give 2 `.txt` files in which the image paths are listed, and with that this script can safely compare the corresponding image pairs. For more details, please check the script.

You can omit the `-o` option (a filename which the computed LPIPS for each image pair is recorded), then all the LPIPS score are displayed in terminal. Also, you can choose `VGG` as the feature extractor by specifying it with `--model` option.


## Reference

* Paper: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/pdf/1801.03924.pdf)
* Github: [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
