# Fine-tuning the YOLO v2 Network with a subset of MSCOCO dataset
This tutorial explains how to fine tune YOLO v2 network using a custom data set. In this tutorial, we use a subset of MSCOCO dataset to demonstrate how to finetune YOLO v2 network with a smaller dataset. Our YOLO v2 training scripts rely on the same dataset structure as what is used by [original authors](https://pjreddie.com/darknet/yolov2/). So, to finetune, create your own custom dataset in a format described below.
## Downloading MSCOCO dataset
From the official [MSCOCO website](http://cocodataset.org/#download), download the following files:
* 2014 Train images
* 2014 Val images
* 2014 Train/Val annotations

Please download, unzip, and place the images in: **coco/images/** and download and place the annotations in: **coco/annotations/**. Download [labels](https://pjreddie.com/media/files/coco/labels.tgz), unzip and place it in **coco/labels/**. These label files are not official, and they can be created from the original COCO annotation files. Rename the labels folder as **labels-original**. After the following the above steps the coco data set structure should be similar to as mentioned below:
```
--coco(dir)
----images(sub-dir)
--------train2014
--------val2014
----labels-original(sub-dir)
--------train2014
--------val2014
----annotations(sub-dir)
```
Before going further, clone the repository: [https://github.com/sony/nnabla-examples](https://github.com/sony/nnabla-examples). To skip the detailed explanation mentioned in this tutorial, you can refer [Quick tutorial for finetuning YoloV2 on MSCOCO subset](tutorial_finetuning_mscoco_subset.md) to have a quick overview on the process of finetuning YOLO v2 on MSCOCO subset.
## Creating a subset of MSCOCO data set
The data set for training YOLO v2 consists of three parts:
1. The images
2. The labels for the bounding boxes
3. The class names

Please refer [Tutorial: Training the YOLO v2 Network with a Custom Dataset](tutorial_custom_dataset.md) to see the standard data set format for images, labels and class names required for training YOLO v2. The same format has been followed in this example. 

**Selecting classes for the creating subset of MSCOCO data set:**
* To select the classes for your subset, at first, you should know all the classes names in the MSCOCO data set 
   and their respective class IDs.
* The data set has an annotation directory which will contain files with .json format, you can read this file to know the class names and their respective class IDs in MS COCO dataset.
* Run the following command and note down the class IDs which you want to include while creating the subset. 
 
```python
import json
js = json.load('./instances_val2014.json')
print(js['categories'])

```
The subset of  MSCOCO data set can be created by running the following command in the terminal:
```bash
python finetune/create_mscoco_subset.py -dp {path-to-coco-data set} -sp {path where subset should be created} \
    -it {no. of training images required per category} -iv {no. of validation images required per category} \
    -l {subset class ids}
#Ex.
#python create_mscoco_subset.py --data-path '/home/ubuntu/coco' --subset-path '/home/ubuntu/subset/' \
#    -it 1000 -iv 500 -l 10 73
```
Where, `-dp` is the path of coco dataset, `-sp` is the path where subset will be created, `-it` is the number of images you want in training dataset, `-iv` is the number of images you want in validation dataset and `-l` is selected class IDs which you want to include in subset.
After running the subset creation code, you should see the following files under your subset directory 
```
train2014.txt
val2014.txt 
coco.names
```
In this example, we will create four different types of subsets as listed below for finetuning purpose:
1. Data set with 10 classes and 1000 images per class in training data set
If you have Selected 10 class Ids and want 1000 training images and 500 validation images per class in your data set run the below command:
``` bash  
python finetune/create_mscoco_subset.py -dp {path-to-coco-data set} -sp {path where subset should be created} -it 1000 -iv 500  -l 10 13  22 23 41 47 53 54  61 73
```
2. Data set with 10 classes and 100 images per class in training data set: 
```bash
python finetunecreate_mscoco_subset.py -dp {path-to-coco-data set} -sp {path where subset should be created} -it 100 -iv 50 -l 10 13  22 23 41 47 53 54  61 73
```
3. Data set with single class and 1000 images per class in training data set: 
```bash
python finetune/create_mscoco_subset.py -dp {path-to-coco-data set} -sp {path where subset should be created} -it 1000 -iv 500 -l 73
```
4. Data set with single class and 100 images per class in training data set: 
```bash
python finetune/create_mscoco_subset.py -dp{path-to-coco-data set} -sp {path where subset should be created} -it 100 -iv 50 -l 73
```

## Fine-tuning YOLO v2 on the subset of MSCOCO data set
Now that the subset data set is created, we can finetune YOLO v2 on this subset. The original authors have trained YOLO v2 on variety of data sets and have provided the [pre-trained models](https://pjreddie.com/darknet/yolov2/). In this example, we will be using a pre-trained model which was obtained by training YOLO v2 on VOC dataset. This pre-trained file is in binary format and we need to convert it to nnabla .h5 format. Download the author's [YoloV2 voc weights](https://pjreddie.com/media/files/yolov2-voc.weights) and then convert the weights to nnabla .h5 format by running the below command:
```shell
cd nnabla-examples/object-detection/yolov2
python convert_yolov2_weights_to_nnabla.py --input {path to weights downloaded above} \
    --classes {No. of classes in the created dataset}
```
The above command creates a weight file named as `yolov2.h5`. For fine tuning purpose, this weight file will be used as the initial weight and all the layers of the YOLO v2 network will be trained on the subset data set. We are ready now to finetune the YOLO v2 network on created subset. Run the below commands in the terminal:

 ```
 cd nnabla-examples/object-detection/yolov2
   #for data set type 1,
 python train.py -w ./yolov2.h5 -t {path-to-dataset}/train2014.txt --max-batches 14000 --steps "6000,10000" --burn-in 500  
 --num-classes {no.of classes} -a coco -o backup -g {GPU number} --fine-tune
 ex: python train.py -w ./yolov2.h5 -t /home/ununtu/subset/train2014.txt --max-batches 14000 --steps "6000,10000" --burn-in 500 --num-classes 10 -a coco -o backup -g 0 --fine-tune
   #for data set type 2
 python train.py -w ./yolov2.h5 -t {path-to-data set}/train2914.txt --max-batches 2500 --steps "1000,2000" --burn-in 400
 --num-classes {no.of classes} -a coco -o backup -g {GPU number} --fine-tune
   #for data set type 3
 python train.py -w ./yolov2.h5 -t {path-to-data set}/train2014.txt --max-batches 2500 --steps "1000,2000" --burn-in 400
 --num-classes {no.of classes} -a coco -o backup -g {GPU number} --fine-tune
   #for data set type 4
 python train.py -w ./yolov2.h5 -t {path-to-data set}/train2014.txt --max-batches 350 --steps "100,200" --burn-in 100
 --num-classes {no.of classes} -a coco -o backup -g {GPU number} --fine-tune
 ```
   * -w specifies the location of weight file.
   * -t option must specify the location of text file containing path of training images
   * --max-batches determines the number of epoch, please see [here](https://github.com/sony/nnabla-examples/blob/master/object-detection/yolov2/train.py#L200) for more detail. --steps and --burn-in are the techniques used for changing learning rates over epochs. **For finetuning pupose, we have adjusted these 3 hyperparameters, in this tutorial.**
   * --num-classes is number of classes in the subset, -a is for selecting anchor biases and -g should specify which gpu device should be utilized.
   * -o specifies the location of directory were weight files are saved during finetuning.
 After the training is complete we can evaluate the performance of the model by calculating the mean average precision(mAP).
### mAP Evaluation
Mean Average Precision is the score used for evaluating the image object detection performance in the original YOlO v2 paper. The steps for evaluating YOLO v2 model trained on MSCOCO data set are mentioned below:
1. The first step of evaluating the mAP for the trained network is to run "valid.py". To do this, run the following command on your terminal:
  ```bash
  python valid.py -w {path to the weight file}.h5 -v {path-to-data set} -o results --num-classes={number of created class} \
      --names {path to category names of the subset} -a coco -g {GPU ID}
 #Ex,python valid.py -w .backup/000090.h5 -v ./subset/val2014.txt -o results --num-classes=10 --names ./subset/coco.names -a coco -g 0  
  ```
  * The -v option must specify location of text file containing path of validation images.
  * -w is path to the weight file obtained after finetuning.
  * After running this on the terminal, valid.py will produce ".txt" files  under the directory results. Each of these text files has the name of the format comp4_det_test_*.txt.   
2. The second step is to create annotation files in VOC format. We will have to create annotation files similar to that of VOC data set by typing following commands in the terminal:
   ```shell
   pip install cytoolz 
   python finetune/coco_to_voc.py --anno-file {path-to-annotation-file-for-MSCOCO}/instances_val2014.json --output-dir {path-to-create-xml-files}
   ```
  * This code has been forked from (https://github.com/CasiaFan/Dataset_to_VOC_converter/blob/master/anno_coco2voc.py)
  * --anno_file should specify the location of original annotation file provided by MSCOCO data set.
  * --output_dir should specify the location where the xml file should be created.
  * This command will produce xml files for each image name (ex.COCO_val2014_000000001153.xml).
3. The Last step is to run the following command on terminal to get the mAP of the network:
  ```bash
   python scripts/voc_eval.py --anno-dirpath {path-to-output-directory-created-in-step2}/ -d {path-to-validation-data set} \
       --dataset-classes-name {path-to-class-names} \
       --dataset-name {MSCOCO} --res-prefix {path-to-results-directory-created-in step1}/comp4_det_test_
   ```
  * --anno-dirpath specifies the location of output directory which contains xml files created in step2
  * -d will have the location of the .txt file which will have the location of all validation images. This will be same as used in step1
  * --dataset-classes-name specifies the location of class names given by MSCOCO data set as coco.names.
  * --dataset-name specifies which data set has been used for training purpose, currently only 'MSCOCO' and 'VOC' is supported
  * This script will output data in the following format:
  
```
AP for traffic light = 0.{number}
AP for stop_sign = 0.{number}    
AP for elephant = 0.{number}     
............................    
............................    
For all classes.                 
Mean AP = 0.{number}             
```

  * Here, each 0.[number] in this example will be outputted as an actual number. Each line in the output, such as "AP for aeroplane," indicates the Average Precision (AP) for the given class name, here, "aeroplane." The final line , "Mean AP = 0.[number]," is the Mean Average Precision(mAP) score for the network.
  * After this output, the script will also output a list of numbers
  
```
0.{number}
0.{number}
......... 
..........
Mean AP = 0.{number}
```
  * The contents of this output are basically the same as the first half, except the text is excluded, and the number is rounded to the third decimal place.
  
## Quick tutorial for finetuning YOLO v2 on MSCOCO subset
This section describes the finetuning pricess in brief with sample commands:
1. Create a subset dataset with 10 classes and 1000 training images per class:
```bash
cd nnabla-examples/objet-detection/yolov2
python finetune/create_mscoco_subset.py -dp /{path-to-coco-dataset}/coco -sp ./subset_10_100 -it 1000 -iv 500  -l 10 13  22 23 41 47 53 54  61 73
```
2. Convert the author's pre-trained model to nnabla H5 format:
```bash
python convert_yolov2_weights_to_nnabla.py --input ./yolov2.weights --classes 10
```
3. Finetune YoloV2 on the created subset:
```bash
python train.py -w ./yolov2.h5 -t ./subset_10_1000/train2014.txt --max-batches 14000 --steps "6000,10000" --burn-in 500 --num-classes 10 -a coco -o backup_10_1000 -g 0 --fine-tune
```
4. Create validation results:
```bash
python valid.py -w .backup_10_1000/000090.h5 -v ./subset_10_1000/val2014.txt -o results_10_1000 --num-classes=10 --names ./subset_10_1000/coco.names -a coco -g 0
```
5. Convert MSCOCO annotation files in VOC format:
```
python finetune/coco_to_voc.py --anno-file {path-to-annotation-file-for-MSCOCO}/instances_val2014.json --output-dir mscoco_annot
```
6. Evaluate mAP:
```
python scripts/voc_eval.py --anno-dirpath mscoco_annot/ -d ./subset_10_1000/val2014.txt --dataset-classes-name ./subset_10_1000/coco.names --dataset-name MSCOCO --res-prefix results_10_1000/comp4_det_test_
```
