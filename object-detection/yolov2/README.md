# YOLO v2 : Image object detection

## Training
The training code was forked from [https://github.com/marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2). See the License section for details.

### Prepare the dataset
Follow the steps in "Training YOLO on VOC" from https://pjreddie.com/darknet/yolov2/ .

### Download the pretrained parameters
Follow the steps in "Download Pretrained Convolutional Weights" from https://pjreddie.com/darknet/yolo/ .

### Convert the pretrained parameters
Run the following:

```bash
python convert_darknet19_448_conv_23_weights_to_nnabla.py --input darknet19_448.conv.23
```

### Train
Run the following:

```bash
python train.py -w ./darknet19_448.conv.23.h5 -o backup
```

For details on optional arguments, run the following command, or see inside `utils.py` .

```bash
python train.py --help
```

Currently, the logs for the loss function values do not get saved by default. If you are working in a Linux environment, use `tee`, such as `python train.py ... | tee log.txt` to save the output log.

### Evaluate the mAP
Run the following:

```bash
python valid.py -w backup/000310.h5 -o results
python2 scripts/voc_eval.py results/comp4_det_test_
```

Edit the `results` argument to change the output directory of the preprocessed calculations. For more details on optional arguments, run the following command, or see inside valid.py.

```bash
python valid.py --help
```


## Inference

TODO: Add requirements.txt

This example demonstrates YOLO v2 object detection inference on NNabla with pretrained weights available at [the original author's website](https://pjreddie.com/darknet/yolo/).

There are two examples that run the object detection model. One is on Python API, another is on ROS C++ node.

### On Python API

This reqruires Python OpenCV. We would recommend you to install Python OpenCV to your Python by building from source using CMake command.

1. Download the darknet model file from the web site.
```
python download_darknet_yolo.py
```

1. Convert the Darknet weight file to NNabla weights.
```
python convert_yolov2_weights_to_nnabla.py
```

1. Run detection. It runs YOLOv2 trained on MS COCO dataset given an image (`dog.jpg`), and outputs an image with bounding boxes `detect.dog.jpg` to the current folder.
```
python yolov2_detection.py [-c cudnn] dog.jpg
```

## On ROS C++ node

NOTE: See [this page](https://github.com/sony/nnabla/tree/master/doc/build/README.md) for a build instruction of C++ libraries.

1. Generate NNP format file for YOLOv2 inference. (Require the weight file created above.)
```shell
python yolov2_nnp.py
```
1. Copy the generated `coco.names` and `yolov2.nnp` to `./ros/nnabla_object_detection/data/`.
1. Create a symbolic link to `nnabla_object_detection` at your catkin_workspace.
1. Build your catkin workspace. The headers and so files of nnabla, nnabla_utils and nnabla_ext_cuda must be in paths. If you don't use a CUDA extension of NNabla, add `-DWITH_CUDA=OFF` to `catkin_make` command.
1. Launch `roslaunch nnabla_object_detection demo.launch` with appropriate args. See the launch file for options.

---
## License
`dataset.py`, `image.py`, `region_loss.py`, `train.py`, `utils.py`, and `valid.py` were forked from  [https://github.com/marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2), licensed under the MIT License (see [./LICENSE.external](./LICENSE.external) for more details).

`scripts/voc_eval.py` was forked from [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), licenced under the MIT License (see [./LICENSE.external](./LICENSE.external) for more details).