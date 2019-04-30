# ImageNet classification example

For using the [DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html) data iterator, the procedure to prepare the dataset and scripts are basically same.

NOTE: Up to now, DALI is the beta version. If it graduates from the beta version, NNabla DALI iterator will be integrated to NNabla itself.


## Preparation

### DALI installation


Install NVIDIA DALI first, 

- CUDA 9.0
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
```

- CUDA 10.0
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali
```


### Datasets

* [training dataset](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar) (this is the same one described in [README.md](./README.md))
* [validation dataset](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar) (this is the same one described in [README.md](./README.md))

Then, untar the two files.

```bash
tar zxvf ILSVRC2012_img_train.tar
tar zxvf ILSVRC2012_img_val.tar
```


### Labeled image file list

The labeled image file list necessary for using DALI data iterator for the typical classification problem. Run the following command for producing.

```bash
python dali_create_label.py -T <path/to/ILSVRC2012_img_train>
```

Now, you find the `train_label` and `val_label` under the working directory, each of which looks like, 

- train_label
```
n02172182/n02172182_4324.JPEG 625
n02172182/n02172182_6601.JPEG 625
n02172182/n02172182_1015.JPEG 625
n02172182/n02172182_3296.JPEG 625
...
```

- val_label
```
ILSVRC2012_val_00000001.JPEG 490
ILSVRC2012_val_00000002.JPEG 361
ILSVRC2012_val_00000003.JPEG 171
...
```

## Training

Run the following code for training ResNet-50 with 4 GPUs.


```bash
mpirun -n 4 python dali_multi_device_multi_process_classification.py -b 32 -a 2 -L 50 -l 0.1 \
  -i 2000000 -v 20004 -j 1563 -s 20004 \
  -D 600000,1200000,1800000 \
  -T <path/to/ILSVRC2012_img_train> -TL train_label \
  -V <path/to/ILSVRC2012_img_val/tmpdir/> -VL val_label \
  -N 4
```


## Validation using single GPUs

Run the following code for training ResNet-50 with a single GPU.

```bash
python dali_validation.py -d 0 -b 50 -L 50 -V <path/to/ILSVRC2012_img_val/tmpdir/> -VL val_label \
    --model-load-path <path/to/modelfile>
```








