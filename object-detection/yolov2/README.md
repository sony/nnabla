# YOLO v2 : Image object detection

Welcome to YOLO-v2-NNabla! YOLO-v2-NNabla allows you to run image object detection on NNabla, producing an image like what is shown below.

For a quick start on running image object detection on a given image using pre-trained network weights, please see the [quick start guide](./quickstart.md)!

(TODO: insert image)

- For details on training and evaluating your network's mAP (Mean Average Precision), see [Tutorial: Training the YOLO v2 Network with YOLO-v2-NNabla](./tutorial/tutorial_training.md).
- For details on training and evaluating your network using your own image feature extractor, see [Tutorial: Training the YOLO v2 Network with a Custom Feature Extractor](./tutorial/tutorial_custom_features.md).
- For details on training and evaluating your network using your own dataset, see [Tutorial: Training the YOLO v2 Network with a Custom Dataset](./tutorial/tutorial_custom_dataset.md).


## Running Image Object Detection
YOLO-v2-NNabla currently runs on Python, or on ROS C++ nodes.

### On Python
For instructions on running image object detection the Python API, please see the [quick start guide](./quickstart.md). This tutorial covers the instructions on how to run image object detection on a given image, using pre-trained network weights.

### On ROS C++ nodes

NOTE: See [this page](https://github.com/sony/nnabla/tree/master/doc/build/README.md) for a build instruction of C++ libraries.

1. Generate NNP format file for YOLOv2 inference. (Require the weight file created above.)
```shell
python yolov2_nnp.py
```
1. Copy the generated `coco.names` and `yolov2.nnp` to `./ros/nnabla_object_detection/data/`.
1. Create a symbolic link to `nnabla_object_detection` at your catkin_workspace.
1. Build your catkin workspace. The headers and so files of nnabla, nnabla_utils and nnabla_ext_cuda must be in paths. If you don't use a CUDA extension of NNabla, add `-DWITH_CUDA=OFF` to `catkin_make` command.
1. Launch `roslaunch nnabla_object_detection demo.launch` with appropriate args. See the launch file for options.


## Training, Evaluating, and Detection Using Trained Parameters
The training code was forked from [https://github.com/marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2). See the License section for details.

For details on training and evaluating your network's mAP (Mean Average Precision), see [Tutorial: Training the YOLO v2 Network with YOLO-v2-NNabla](./tutorial/tutorial_training.md).

---
## License
`dataset.py`, `image.py`, `region_loss.py`, `train.py`, `utils.py`, and `valid.py` were forked from  [https://github.com/marvis/pytorch-yolo2](https://github.com/marvis/pytorch-yolo2), licensed under the MIT License (see [./LICENSE.external](./LICENSE.external) for more details).

`scripts/voc_eval.py` was forked from [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), licensed under the MIT License (see [./LICENSE.external](./LICENSE.external) for more details).