# Quick Start: Image Object Detection with YOLO-v2-NNabla
Welcome to YOLO-v2-NNabla! This tutorial will explain in detail on how to run image object detection using YOLO-v2-NNabla.


## Prerequisites: Install NNabla and Other Required Software
YOLO-v2-NNabla requires the following software and Python packages installed in your system:

- Python
  - OpenCV Python package (This is **optional** for better drawing of detected bounding boxes. PIL is used if it is not available on your system.)
- CUDA
- cuDNN

### Installing Python and NNabla
Details on installing NNabla is explained in [the NNabla Documentation](https://nnabla.readthedocs.io/en/latest/python/installation.html).

## Step 1: Clone This Repository
If you have git installed on your system, simply run:
```
git clone https://github.com/sony/nnabla-examples/
```
This will clone the whole nnabla-examples repository, including YOLO-v2-NNabla, to your system.

If you don't have git installed on your system, please access the [GitHub repository for nnabla-examples](https://github.com/sony/nnabla-examples/). After you have opened the nnabla-examples page, please click on the "Clone or Download" button, which will allow you to download the whole nnabla-examples repository, including YOLO-v2-NNabla, as a zip archive.


## Step 2: Download the Pretrained Network Parameters and Category Descriptions
The original author of the YOLO v2 paper has published several versions of pretrained YOLO v2 pretrained network in the original [YOLO v2 project website](https://pjreddie.com/darknet/yolov2/). To download the pretrained network parameters from this website, run the following on your terminal:
```
python download_darknet_yolo.py
```
Running this script will download the following three files from the YOLO v2 project website and the author's GitHub repository:
- Pretrained network weights: https://pjreddie.com/media/files/yolov2.weights
  - The network is trained on MS COCO, which has 80 classes. Using these weights will allow you to perform object detection on 80 classes.
  - Downloaded from the YOLO v2 project website.
- Category names: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
  - Describes the names of the 80 classes in the MS COCO dataset.
  - Downloaded from the YOLO v2 paper author's GitHub repository.
- Example image: 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg'
  - Downloaded from the YOLO v2 paper author's GitHub repository.

### Using Other Pretrained Network Weights
Other various versions of pretrained network weights can be manually downloaded from the original YOLO v2 project website. As the time of this writing, the weights can be found in the table at the top of the page, under the row "weights." There are several types of weights available on the page. Currently, the following two weights can be used in YOLO-v2-NNabla:

- https://pjreddie.com/media/files/yolov2.weights
- https://pjreddie.com/media/files/yolov2-voc.weights

These weights are trained for different networks with different structures, trained on different datasets. The first network can detect 80 classes, and the second network can detect 20 classes. This is due to the fact that the first network is trained on the MS COCO Dataset, which has 80 classes, and the second network is trained on the Pascal VOC 2007+2012 dataset (which is an aggregated dataset combined by the author of the paper), which is an object detection dataset with 20 classes.


## Step 3: Convert the network parameters
Once you have downloaded the network parameters, the parameters must be converted to NNabla format. The original files with the \*.weights extension must be converted to the \*.h5 format which can be read by NNabla. For conversion, we will be using a converter script. Run the following code on the terminal, according to the weight file you have chosen:

- yolov2.weights :
  ```
  python convert_yolov2_weights_to_nnabla.py --input yolov2.weights
  ```
- yolov2-voc.weights :
  ```
  python convert_yolov2_weights_to_nnabla.py --input yolov2-voc.weights --classes 20
  ```

After running these conversion scripts, you will get an \*.h5 file, named yolov2.h5 or yolov2-voc.h5, depending on the weight file that you have chosen. We will be using this file later for running object detection.

**Troubleshooting:**
- If you have some issues on converting the weights, check the filenames for the \*.weights file, and make sure you have downloaded the correct files mentioned in the previous section. A common issue happens yolov2-voc.weights without the required arguments mentioned above. These arguments are required for yolov2-voc.weights since the converting script is set to convert yolov2.weights by default.


## Step 4: Run the Detection Script
Once you have prepared the weights, prepare an image to run the detection on!

Save the image with the name `input_image.jpg`. You can then run the following command on your terminal to run object detection:
```
python yolov2_detection.py input_image.jpg
```

If you are using yolov2-voc.h5, use the following arguments:
```
python yolov2_detection.py input_image.jpg \
--weights yolov2-voc.h5 \
--class-names ./data/voc.names \
--classes 20 \
--anchors voc
```
- **Remark:** The `--anchors` argument spcifies the anchor box biases by the following format; `--anchors="<w0>,<h0>,<w1>,<h1>,...,<wN>,<hN>"`. As special cases, if a string `voc` or `coco` is specified to `--anchors`, the preset numbers described in `arg_utils.py:get_anchors_by_name_or_parse` are used. These numbers represent a fixed parameter determined at training-time, which is dependent on the dataset. By default, `yolov2_detection.py` expects to use yolov2.h5, which are weights that are trained on the MS COCO dataset. The corresponding `--anchors` argument `coco` is set by default by `yolov2_detection.py`. On the other hand, when using yolov2-voc.h5, you must specify `voc`. Although `yolov2_detection.py` expects MS COCO as the default dataset, the training script and the validation script, `train.py` and `valid.py` respectively, expects Pascal VOC 2007+2012 as the default dataset, i.e., `coco`.

If your image file has another name, change the `input_image.jpg` part to the image name to detect for that image. This will output an image file named `detect.input_image.jpg`, or something else that matches your input filename. `detect.input_image.jpg` should look like this:

(TODO: create image)

**Troubleshooting:**
- If the weight file cannot be found, make sure you have specified the correct filename for the `--weights` argument.
- Check if you are specifying `--classes` and `--class-names` correctly if you are using yolov2-voc.h5. As mentioned earlier, yolov2.h5 and yolov2-voc.h5 are trained on different datasets with a different number of classes.
- If the output bounding boxes are off, make sure you have specified the `--anchors` argument correctly.


## Other Topics
The following topics are covered in the following documents:

- For details on training and evaluating your network, see [Tutorial: Training the YOLO v2 Network with YOLO-v2-NNabla](./tutorial/tutorial_training.md).
- For details on training and evaluating your network using your own image feature extractor, see [Tutorial: Training the YOLO v2 Network with a Custom Feature Extractor](./tutorial/tutorial_custom_features.md).
- For details on training and evaluating your network using your own dataset, see [Tutorial: Training the YOLO v2 Network with a Custom Dataset](./tutorial/tutorial_custom_dataset.md).
- For more details on other topics, see the [README](./README.md).
