# Tutorial: Training the YOLO v2 Network with a Custom Dataset

The dataset for training YOLO v2 consists of three parts:

- The images
- The labels for the bounding boxes
- The class names

If possible, please first follow "Step 1: Prepare the Dataset" in [Tutorial: Training the YOLO v2 Network with YOLO-v2-NNabla](./tuorial_training.md), and see what kind of structure the dataset follows. Then, the custom dataset shall be created in the same format as the dataset obtained in that section. This tutorial will cover the format of these three components of the dataset in detail ... (TODO)


## Calculating the Anchor Box Biases
The bias values represent the width-height pairs of a default-size box, called "anchor boxes" in the original YOLO v2 paper (or YOLO v1? TODO:cite). These width-height pairs represent the most common box sizes that appear in the training dataset. In the original paper, the bias values are determined by running k-means clustering (with instructions given later) on the bounding boxes.

As mentioned in [Quick Start: Image Object Detection with YOLO-v2-NNabla](../quickstart.md), the bias values (passed by the `--biases` argument in `yolov2_detection.py`, and `--anchors` argument in `train.py`) are tailored for the specific dataset that was used at training time. The values used in the aforementioned tutorial is provided by the author of the YOLO v2 paper (TODO:cite), and are the biases for the MS COCO Dataset (or the VOC 2007+2012 dataset).

Thererore, when using a custom dataset, it is best to calculate the bias values for the anchor boxes tailored for the dataset. However, it is also not impossible to reuse the anchor box sizes provided in the code, although this could be a compromise for the performance. If you wish to calculate your own bias values, please proceed to the next section. If you wish to use the default bias values provided inside YOLO-v2-NNabla, please proceed to the second next section.

### Calculating the Bias Values
This section will describe how to calculate the bias for the custom dataset.

The distance metric between two given boxes, used in the k-means clustering, is given as follows:
1. Translate the boxes so that they share the center points.
2. Calculate the IoU (Intersection over Union) index is used for the distance metric between boxes. The IoU calculated this way is used as the distance metric between two boxes.

### Using the Bias Values at Training and Inference Time
Make sure the same bias values are used during training and during inference.

(TODO)


## The Image Format
(TODO)

Please follow the filename format... (TODO)


## The Label Format
(TODO)

It is a list of floats normalized by the image width and height... (TODO)

Please follow the filename format... (TODO)


## The Class Name Format
This is straightforward - simply list the names delimited by a newline in a single text file...
(TODO)
