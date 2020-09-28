# Neural Network Libraries - Examples

This repository contains working examples of [Neural Network Libraries](https://github.com/sony/nnabla/).
Before running any of the examples in this repository, you must install the Python package for Neural Network Libraries. The Python install guide can be found [here](https://nnabla.readthedocs.io/en/latest/python/installation.html).

Before running an example, also run the following command inside the example directory, to install additional dependencies:

```
cd example_directory
pip install -r requirements.txt
```


## Docker workflow

* Our Docker workflow offers an easy installation and setup of running environments of our examples.
* [See this page](doc/docker.md).


## Interactive Demos

We have prepared interactive demos, where you can play around without having to worry about the codes and the internal mechanism. You can run it directly on [Colab](https://colab.research.google.com/) from the links in the table below.


| Name        | Notebook           | Task  |
|:------------------------------------------------:|:-------------:|:-----:|
| [ESR-GAN](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/esrgan.ipynb) | Super-Resolution|
| [Self-Attention GAN](http://proceedings.mlr.press/v97/zhang19d/zhang19d.pdf)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/sagan.ipynb) | Image Generation|
| [Face Alignment Network](https://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/fan.ipynb) | Facial Keypoint Detection |
| [PSMNet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Pyramid_Stereo_Matching_CVPR_2018_paper.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/psmnet.ipynb) | Depth Estimation |
| [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)/[ResNeXt](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)/[SENet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/imagenet_classification.ipynb) | Image Classification |
| [YOLO v2](https://openaccess.thecvf.com/content_cvpr_2017/papers/Redmon_YOLO9000_Better_Faster_CVPR_2017_paper.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/yolov2.ipynb) | Object Detection |
| [StarGAN](https://arxiv.org/abs/1711.09020) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/stargan.ipynb) | Image Translation |
| [MixUp](https://openreview.net/pdf?id=r1Ddp1-Rb) / [CutMix](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) / [VH-Mixup](https://arxiv.org/pdf/1805.11272.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/dataaugmentation.ipynb) | Data Augmentation |
| [StyleGAN2](https://arxiv.org/pdf/1912.04958.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/stylegan2.ipynb) | Image Generation |
