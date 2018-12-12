# Self-Attention Generative Adversarial Networks

This is the reproduction work of *Self-Attention Generative Adversarial Networks*.


<img class="alignnone size-medium wp-image-328" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110034/000985-300x300.png" alt="" width="300" height="300" /><img class="alignnone size-medium wp-image-329" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110037/000949-300x300.png" alt="" width="300" height="300" />
<img class="alignnone size-medium wp-image-330" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110039/000937-300x300.png" alt="" width="300" height="300" /><img class="alignnone size-medium wp-image-331" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110042/000933-300x300.png" alt="" width="300" height="300" />
<img class="alignnone size-medium wp-image-332" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110045/000725-300x300.png" alt="" width="300" height="300" /><img class="alignnone size-medium wp-image-333" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110050/000607-300x300.png" alt="" width="300" height="300" />
<img class="alignnone size-medium wp-image-334" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110054/000574-300x300.png" alt="" width="300" height="300" /><img class="alignnone size-medium wp-image-335" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110057/000323-300x300.png" alt="" width="300" height="300" />
<img class="alignnone size-medium wp-image-336" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110103/000153-300x300.png" alt="" width="300" height="300" /><img class="alignnone size-medium wp-image-337" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04110107/000096-300x300.png" alt="" width="300" height="300" />

<img class="alignnone size-full wp-image-326" src="https://blog.nnabla.org/wp-content/uploads/2018/12/04105043/categorical-morphing.gif" alt="" width="128" height="128" />


Self-Attention Generative Adversarial Networks (SAGAN) is a good GAN example since it contains many recently proposed techniques for training GANs, the following techniques are included in this SAGANs example.


- Spectral normalization: normalization of weights with its singular value to constrain the discriminator Lipchitz norm
- Two Time-Scale Update Rule (TTUR): unbalanced learning rate for the generator and discriminator
- Self-Attention: Attention of the whole region of an image
- cGANs with Projection Discriminator: projection of labels in the discriminator at the end, then use it in the adversarial loss
- Class Conditional Batch Normalaization (CCBN): Incorporation of label information into the gamma and beta of the batch normalization.


# Dataset

Download imagenet dataset [here](https://imagenet.herokuapp.com/), then untar *ILSVRC2012_img_train.tar* and run the following script to resize all images in advance.

```bash
bash preprocess.sh <path to the top directory of imagenet dataset> <path to save dir>
```

Top directory of imagenet dataset is one that contains directories each of which represents the class id.

# Pre-trained model

You can download the pre-trained model [here](https://nnabla.org/pretrained-models/nnabla-examples/GANs/sagan/params_999999.h5). If so, you can skip the next section.


# Training

```bash
mpirun -n 4 python train_with_mgpu.py -c cudnn -b 64 \
    -T <path to save dir> \
    -L ./dirname_to_label.txt \
    --monitor-path ./result/example_000 \
    --max-iter 999999 \
    --save-interval 10000
```

Training takes about 9 days up to 1,000,000 iteration and with 256 batch size in the distributed system using NVIDIA DGX Station (4 x Tesla GV100). Generated images seemingly become visually plausible after a few 100 K iteration.


**NOTE** Generated images does not normally collapse for each class when using 64 batch in a worker. If using 64 batch with the gradient accumulation (i.e., 32 batch x 2) in a worker makes a model collapse for some classes partially. It could be considered that we have 1K-class generative model and the batch normalization, so when computing the batch statistics (i.e., the batch mean and variance), the number of classes in the batch statistics is not enough if we use small batch size, e.g., 1K-class vs 32 classes in the batch statistics at most.


# Generation

To generate images with random classes, run like

```bash
python generate.py -c cudnn -d 0 -b 36 \
    --monitor-path ./result/example_000 \
    --model-load-path ./result/example_000/params_999999.h5
```

If you want to see generated images of a specific class, add *class-id* (0-based) option.

```bash
python generate.py -c cudnn -d 0 -b 36 \
    --monitor-path ./result/example_000 \
    --model-load-path ./result/example_000/params_999999.h5 \
    --class-id 153
```

Looking *label_to_classname.txt* in the directory, this [page](http://image-net.org/explore) is helpful to see images visually.


When you want to see images for all classes, run like

```bash
python generate.py -c cudnn -d 0 -b 36 \
    --generate-all \
    --monitor-path ./result/example_000 \
    --model-load-path ./result/example_000/params_999999.h5
```

All images generated are found in *./result/example_000/Generated-Image-\**, and *--truncation-threshold* argument might help to control quality of images and divergence of images.


# Morphing

You can also generate images as the categorical morphing.


```bash
python morph.py -c cudnn -d 0 -b 1 --n-morphs 8 \
    --monitor-path ./result/example_000 \
    --model-load-path ./result/example_000/params_999999.h5 \
    --from-class-id 947 --to-class-id 153
```

All images generated as the between-two-classes morphing are found in *./result/example_000/Morphed-Image-\**.


# Matching

**Not supported officially yet.**

In order to see similar images to an generated image in an embedding space, e.g., the before-affine layer ResNet-50 and after the average pooling layer run this command, 

```bash
python match.py -c cudnn -d 0 \
             -T /data/datasets/imagenet/train_cache_sngan \
             -L ./dirname_to_label.txt \
             --monitor-path ./result/example_000 \
             --model-load-path ./result/example_000/params_999999.h5 \
             --nnp-inception-model-load-path <path to>/pretrained-imagenet/NNP/Resnet-50_4_178.nnp \
             --variable-name AveragePooling \
             --image-size 224 \
             --nnp-preprocess \
             --top-n 15 \
             --class-id 153
```

then, you can see the tiled images which contains an generated image on the top-left and similar real images in an embedding space on the other cell of the tiled images.


# Validation

**Not supported officially yet.**

### Inception Score (w/ Inception-V3)

```bash
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 \
             --monitor-path ./result/example_000 \
             --model-load-path ./result/example_000/params_999999.h5 \
             --nnp-inception-model-load-path <path to>/pretrained-imagenet/NNP/Inception-v3.nnp \
             --image-size 224 \
             --evaluation-metric IS \
             --nnp-preprocess
```

We get ~ 72, it totally varies using different upsampling algorithm.

### Inception Score (w/ ResNet-50)

```bash
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 \
             --monitor-path ./result/example_000 \
             --model-load-path ./result/example_000/params_999999.h5 \
             --nnp-inception-model-load-path <path to>/pretrained-imagenet/NNP/Resnet-50_4_178.nnp \
             --image-size 224 \
             --evaluation-metric IS \
             --nnp-preprocess
```

### Frechet Inception Distance (w/ Inception-V3)

```bash
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 \
             --monitor-path ./result/example_000 \
             --model-load-path ./result/example_000/params_999999.h5 \
             --nnp-inception-model-load-path <path to>/pretrained-imagenet/NNP/Inception-v3.nnp \
             --image-size 320 \
             --evaluation-metric FID --variable-name AveragePooling_2 \
             -V /data/datasets/imagenet/val_data \
             -L /data/datasets/imagenet/dirname_to_label.txt
```

We get ~ 9.5, it totally varies using different upsampling algorithm.

### Frechet Inception Distance (w/ ResNet-50)

```bash
python evaluate.py -c cudnn -d 0 -b 25 --max-iter 400 \
             --monitor-path ./result/example_000 \
             --model-load-path ./result/example_000/params_999999.h5 \
             --nnp-inception-model-load-path <path to>/pretrained-imagenet/NNP/Resnet-50_4_178.nnp \
             --image-size 224 \
             --evaluation-metric FID --variable-name AveragePooling \
             -V /data/datasets/imagenet/val_data \
             -L /data/datasets/imagenet/dirname_to_label.txt \
             --nnp-preprocess
```


When you set the directory which contains h5 parameter files to *--model-load-path*, the evaluation script computes the metrics to all results in order of the saved history.


# References (paper)
1. Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena, "Self-Attention Generative Adversarial Networks", https://arxiv.org/abs/1805.08318
2. Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida, "Spectral Normalization For Generative Adversarial Networks", https://arxiv.org/abs/1802.05957
3. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", https://arxiv.org/abs/1706.08500
4. Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He, "Non-local Neural Networks", https://arxiv.org/abs/1711.07971
5. Takeru Miyato, Masanori Koyama, "cGANs with Projection Discriminator", https://arxiv.org/abs/1802.05637
6. Andrew Brock, Jeff Donahue, Karen Simonyan, "Large Scale GAN Training for High Fidelity Natural Image Synthesis", https://arxiv.org/abs/1809.11096

# References (code)
1. https://github.com/pfnet-research/sngan_projection
2. https://github.com/rosinality/sagan-pytorch
3. https://github.com/brain-research/self-attention-gan
