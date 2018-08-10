# Quick Start: Image Object Detection with YOLO-v2-NNabla
Welcome to YOLO-v2-NNabla! This tutorial will explain in detail on how to run image object detection using YOLO-v2-NNabla.


## Prerequisites: Install NNabla and Other Required Software
YOLO-v2-NNabla requires the following software and Python packages installed in your system:

- Python
  - OpenCV 2 (`cv2`) package
- CUDA
- cuDNN

### Installing Python and NNabla
Details on installing NNabla on Linux is explained in [the NNabla Documentation](https://nnabla.readthedocs.io/en/latest/python/install_on_linux.html). If you are running on Windows, details on installing NNabla on Windows is can be found in [the NNabla Documentation](https://nnabla.readthedocs.io/en/latest/python/install_on_windows.html). This tutorial will cover a summary of the steps that are relevant on running YOLO-v2-NNabla on Windows.

Following the documentation, please first install the following software on your system:

- Python
- CUDA
- cuDNN

Next, check the version for CUDA and cuDNN installed on your system. As of the time of this writing, NNabla is currently compatible with the following combinations of CUDA and cuDNN versions:

- nnabla-ext-cuda80 (CUDA 8.0 x cuDNN 7.1)
- nnabla-ext-cuda90 (CUDA 9.0 x cuDNN 7.1)
- nnabla-ext-cuda91 (CUDA 9.1 x cuDNN 7.1)
- nnabla-ext-cuda92 (CUDA 9.2 x cuDNN 7.1)

Once you have checked the CUDA version on your system, NNabla can be installed using `pip` by the following commands:
```
pip install -u nnabla
pip install -u nnabla-ext-cuda90
# Please replace `nnabla-ext-cuda90`
# according to the CUDA and cuDNN version installed in your system!
```

### Installing OpenCV 2
(TODO)


## Step 1: Clone This Repository
If you have git installed on your system, simply run:
```
git clone https://github.com/sony/nnabla-examples/
```
This will clone the whole nnabla-examples repository, including YOLO-v2-NNabla, to your system.

If you don't have git installed on your system, please access the [GitHub repository for nnabla-examples](https://github.com/sony/nnabla-examples/). After you have opend the nnabla-examples page, please click on the "Clone or Download" button, which will allow you to download the whole nnabla-examples repository, including YOLO-v2-NNabla, as a zip archive.


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
  python convert_yolov2_weights_to_nnabla.py --input yolov2-voc.weights --header 5 --classes 20
  ```
  - **Remark:** The `--header 5` argument is required due to the fact that:
    - The \*.weights has a header that must be skipped when reading the stored float values.
    - The header byte size is different for yolov2.weights and yolov2-voc.weights.

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
--biases 1.3221 1.73145 3.19275 4.00944 \
         5.05587 8.09892 9.47112 4.84053 11.2364 10.0071
```
- **Remark:** The `--biases` argument is may be tricky when using yolov2-voc.h5. These numbers represent a fixed parameter determined at training-time, which is dependent on the dataset. By default, `yolov2_detection.py` expects to use yolov2.h5, which are weights that are trained on the MS COCO dataset. The corresponding `--biases` argument is set by default by `yolov2_detection.py`. On the other hand, when using yolov2-voc.h5, you must specify these parameters manually. Since these parameters depend on the dataset that is being used (to be precise, it is determined by which fixed parameters were used when the training was done for the given weights), the `--biases` corresponding to yolov2-voc.h5 must be manually specified. These values are taken from [line 369 from `utils.py`](https://github.com/sony/nnabla-examples/blob/master/object-detection/yolov2/utils.py#L369). Although `yolov2_detection.py` expects MS COCO as the default dataset, the training script, `train.py`, expects Pascal VOC 2007+2012 as the default dataset. Therefore, using the `--biases` from `utils.py` (which is the utility function file for `train.py`) will allow you to specify the correct `--biases` (which is called `--anchors` in `utils.py`) for yolov2-voc.h5.

If your image file has another name, change the `input_image.jpg` part to the image name to detect for that image. This will output an image file named `detect.input_image.jpg`, or something else that matches your input filename. `detect.input_image.jpg` should look like this:

(TODO: create image)

**Troubleshooting:**
- If the weight file cannot be found, make sure you have specified the correct filename for the `--weights` argument.
- Check if you are specifying `--classes` and `--class-names` correctly if you are using yolov2-voc.h5. As mentioned earlier, yolov2.h5 and yolov2-voc.h5 are trained on different datasets with a different number of classes.
- If the output bounding boxes are off, make sure you have specified the `--biases` argument correctly. Make sure that there are 10 numbers (this is the same for both yolov2.h5 and yolov2-voc.h5).


## Other Topics
The following topics are covered in the following documents:

- For details on training and evaluating your network, see [Tutorial: Training the YOLO v2 Network with YOLO-v2-NNabla](./tutorial/tutorial_training.md).
- For details on training and evaluating your network using your own image feature extractor, see [Tutorial: Training the YOLO v2 Network with a Custom Feature Extractor](./tutorial/tutorial_custom_features.md).
- For details on training and evaluating your network using your own dataset, see [Tutorial: Training the YOLO v2 Network with a Custom Dataset](./tutorial/tutorial_custom_dataset.md).
- For more details on other topics, see the [README](./README.md).
