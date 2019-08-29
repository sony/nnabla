# Tutorial: Training the YOLO v2 Network with a Custom Dataset

The dataset for training YOLO v2 consists of three parts:

- The images
- The labels for the bounding boxes
- The class names

If possible, please first follow "Step 1: Prepare the Dataset" in [Tutorial: Training the YOLO v2 Network with YOLO-v2-NNabla](./tuorial_training.md), and see what kind of structure the dataset follows. Then, the custom dataset shall be created in the same format as the dataset obtained in that section. This tutorial will cover the format of these three components of the dataset in detail.


## Calculating the Anchor Box Biases
The bias values represent the width-height pairs of a default-size box, called "anchor boxes" in [the original YOLO v2 paper][1]. These width-height pairs represent the most common box sizes that appear in the training dataset. In the original paper, the bias values are determined by running k-means clustering (with instructions given later) on the bounding boxes.

As mentioned in [Quick Start: Image Object Detection with YOLO-v2-NNabla](../quickstart.md), the bias values (passed by the `--anchors` argument in `yolov2_detection.py`, `train.py` and `valid.py`) are tailored for the specific dataset that was used at training time. The values used in the aforementioned tutorial is provided by the author of [the YOLO v2 paper][1], and are the biases for the MS COCO Dataset (or the VOC 2007+2012 dataset).

Therefore, when using a custom dataset, it is best to calculate the bias values for the anchor boxes tailored for the dataset. However, it is also not impossible to reuse the anchor box sizes provided in the code, although this could be a compromise for the performance. If you wish to calculate your own bias values, please proceed to the next section. If you wish to use the default bias values provided inside YOLO-v2-NNabla, please proceed to the second next section.

### Calculating the Bias Values
This section will describe how to calculate the bias for the custom dataset.

The distance metric between two given boxes, used in the k-means clustering, is given as follows:
1. Translate the boxes so that they share the center points.
2. Calculate the IoU (Intersection over Union) index is used for the distance metric between boxes. The IoU calculated this way is used as the distance metric between two boxes.

### Using the Bias Values at Training and Inference Time
Make sure the same bias values are used during training and during inference.

(TODO)


## The Image Format
Darknet model takes training and validation images as input by reading their paths from text file. So for this, Darknet needs two text files, one containing path of all the training images and the other containing path of all the validation images. Similarly two text files containing paths of training images and validation images should be created for the custom dataset. The contents of these text files has been depicted below,please follow the filename format:
```
Expected Output:
----train2014.txt(text file inside images folder containing path of training images)
--------------./coco/images/train2014/COCO_train2014_000000236955.jpg
--------------./coco/images/train2014/COCO_train2014_000000203069.jpg and so on...
----val2014.txt (text file inside images folder containing path of validation images)
--------------./coco/images/val2014/COCO_val2014_000000000164.jpg
--------------./coco/images/val2014/COCO_val2014_000000000283.jpg and so on...
```


## The Label Format
The label files required by Darknet are in text file format. Each image in the dataset has its corresponding label file. These label files contain category IDs of the objects present in an image and their ground truth bounding box co-ordinates. The contents of a typical label file is ```<object-class> <x> <y> <width> <height>```  where x, y are co-ordinate of centre point of an object and w,h are width, and height of object are relative to the image's width and height. The same format should be used for label files of the custom dataset. The format of label files has been depicted below, please follow the same format:
```
labels (folder)
----train2014(sub folder with training label files)
--------COCO_train2014_000000000009.txt
----------------45 0.479492 0.688771 0.955609 0.595500 (<category ID> <x> <y> <width> <height>)
----------------45 0.736516 0.247188 0.498875 0.476417
--------COCO_train2014_000000000349.txt
----------------6 0.421352 0.540448 0.842703 0.537062 
----------------58 0.668500 0.665531 0.098250 0.163271
----val2014 (sub folder with validation label files)
--------COCO_val2014_000000000042.txt
----------------16 0.606688 0.341381 0.544156 0.510000 
--------COCO_val2014_000000000626.txt
----------------74 0.520594 0.303323 0.064562 0.088479 
```
## The Class Name Format
This is straightforward - simply list the names delimited by a newline in a single text file. For example, if there are 5 classes in the dataset then their name can be written as shown below:
```
aeroplane
bicycle
bird
boat
bottle
```

[1]: https://arxiv.org/abs/1612.08242 "YOLOv2 arxiv"
