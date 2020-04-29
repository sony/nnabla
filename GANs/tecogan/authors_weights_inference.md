### Inference using pre-trained weights provided by original authors
The pre-trained TecoGAN weight file can be obtained from [here](https://ge.in.tum.de/download/data/TecoGAN/model.zip) which has been provided by the [original authors](https://github.com/thunil/TecoGAN). These pre-trained weight file can directly be used to generate High-Resolution frames from Low-Resolution frames. 
1. From this [link]((https://ge.in.tum.de/download/data/TecoGAN/model.zip)), download the zip file named as `model.zip` and unzip to `./model/` directory.
2. Clone the NNabla examples [repository](https://github.com/sony/nnabla-examples.git).
3. Convert the weight file provided by the author to NNabla's `.h5` format and then use the generate code to produce a HR samples from the given LR samples using the code below:
```
cd nnabla-examples/GANs/TecoGAN
python convert_weights.py --pre-trained-model {path to pre-trained weight} --save-path {path to save the h5 file}
python generate.py --model {path to converted TecoGAN NNabla weight file} --input_dir_LR {input directory} --output_dir {path to output directory}
```
