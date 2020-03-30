# Weights conversion from existing models

Here we convert pretrained weights to nnabla from some existing models provided by original authors.

Available mdoels:

* [SENets](#SENets)

## SENets

We demonstrate how to convert the following models from author's [repository](https://github.com/hujie-frank/SENet) written in Caffe (with some extra user-defined layers).

* SE-ResNet-50
* SE-ResNeXt-50

### Download pretraiend models

You can download pretrained model files for the models listed above from the author's repository. You have to download both `.prototext` (describing network architecture) and `.caffemodel` (storing parameter weights), and later we will use them for weight conversion to nnabla `.h5` files.

### Set up

For ease, we use Docker as an envrionment. With `docker build` command, you can create a docker image with a Dockerfile `Dockerfile.caffe-senet`, which installs all dependencies for the conversion. Please see Docker documententation, if you are not familiar with Docker. It's pretty easy to use.

### Convert weights from Caffe to NNabla

#### SE-ResNet-50

```bash
python senet_converter.py SE-ResNet-50.prototxt SE-ResNet-50.caffemodel se_resnet50.h5
```

#### SE-ResNeXt-50

```bash
python senet_converter.py --resnext SE-ResNeXt-50.prototxt SE-ResNeXt-50.caffemodel se_resnext50.h5
```

### Perform inference

```bash
cd .. # go to a parent folder which contains infer.py

# Use SE-ResNext-50. Specify "se_resnet50" for SE-ResNet-50.
NET_ARCH="se_resnext50"

# Specify the weight file converted above.
WEIGHT_FILE="reference_weight_converter/se_resnext50.h5"

# Perform inference
python infer.py --context='cpu' --arch=${NET_ARCH} {input image file} ${WEIGHT_FILE}
```
