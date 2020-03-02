### Inference using pre-trained weights provided by original authors
The pre-trained ESRGAN and the pre-trained PSNR oriented weightfiles can be obtained from [here](https://drive.google.com/drive/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) which has been provided by the [original authors](https://github.com/xinntao/ESRGAN). These pre-trained weight file can directly be used to do inference on images. See the following [link](./author's_weights_inference.md) to use the original author's pre-trained weights for inference.
1. From this [link]((https://drive.google.com/drive/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY)), download the file named as `RRDB_ESRGAN_x4.pth`.
2. Clone the NNabla examples [repository](https://github.com/sony/nnabla-examples.git).
3. The weights provided by the author's are in`.pth` format, convert the weight file to NNabla `.h5` format and then use the inference code to produce a SR image on a given sample of LR image using the code below:
```
cd nnabla-examples/GANs/esrgan
python convert_weights.py --pre-trained_model {path to pre-trained weights} --save_path {path to save the h5 file}
python inference.py --loadmodel {path to the converted h5 file} --input_image {sample LR image}
```

### Convert VGG19 weights to NNabla
Download the VGG19 weights from [here](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) and then convert these weights to .h5 format using the below code:
```
python convert_vgg19_weights.py --input_file {pre-trained pytorch vgg19 weights} --output_file {path to save the converted model}
```
