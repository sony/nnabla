### Inference using pre-trained weights provided by original authors
The pre-trained TecoGAN weight file can be obtained from [here](https://ge.in.tum.de/download/data/TecoGAN/model.zip) which has been provided by the [original authors](https://github.com/thunil/TecoGAN). These pre-trained weight file can directly be used to generate High-Resolution frames from Low-Resolution frames. 
1. From this [link]((https://ge.in.tum.de/download/data/TecoGAN/model.zip)), download the zip file named as `model.zip` and unzip to `./model/` directory.
2. Clone the NNabla examples [repository](https://github.com/sony/nnabla-examples.git).
3. Convert the weight file provided by the author to NNabla's `.h5` format and then use the generate code to produce a HR samples from the given LR samples using the code below:
```
cd nnabla-examples/GANs/TecoGAN
python convert_tf_tecogan_weights.py --pre-trained-model {path to pre-trained weight} --save-path {path to save the h5 file}
python generate.py --model {path to converted TecoGAN NNabla weight file} --input_dir_LR {input directory} --output_dir {path to output directory}
```
### Convert VGG19 weights to NNabla
Download the VGG19 weights from [here](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) and then convert these weights to .h5 format using the below code:
```
python convert_vgg19_nnabla_tf --pre_trained_model {pre-trained tensorflow vgg19 weights} --save-path {path to save the converted model}
```