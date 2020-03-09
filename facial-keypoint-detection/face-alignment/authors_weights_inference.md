### Inference using pre-trained weights provided by original authors
Author's pre-trained weights can be downloaded from the below links:
### Pre-trained Weights :
| 2D-FAN |  3D-FAN |  ResNetDepth |
|---|---|---|
|[2D-FAN pre-trained weights](https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar)|[3D-FAN pre-trained weights](https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar)|[ResNetDepth pre-trained weights](https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar)|

Clone the nnabla-examples [repository](https://github.com/sony/nnabla-examples.git).
```
cd nnabla-examples/facial-keypoint-detection/face-alignment
```
### Convert the author's pre-trained weights to NNabla format
Run the following command to convert `2D-FAN` / `3D-FAN` pytorch weight to NNabla format.
```
python convert_model_weights.py --pretrained-model {path to downloaded 2D-FAN/3D-FAN weights} --save-path {path to save the .h5 file}
```
Run the following command to convert `ResNetDepth` pytorch weight to NNabla format.
```
python convert_resnet_depth_weights.py --pretrained-model {path to downloaded ResNetDepth weights} --save-path {path to save the .h5 file}
```
### Inference using the converted pre-trained weights.
Run the following command for inference using 2D-FAN
```
python model_inference.py --model {path to converted 2D-FAN NNabla weght file} --test-image {sample image} --output {path to output image}
```
Run the following command for inference using 3D-FAN
```
python model_inference.py --landmarks-type-3D --model {path to converted 3D-FAN NNabla weght file} --resnet-depth-model {path to converted ResNetDepth NNabla weght file} --test-image {path to input sample image } --output {path to output image}
```