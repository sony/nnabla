#!/bin/bash

mkdir -p pretrained_weights/encoder
wget -P pretrained_weights/encoder https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/encoder/weights.h5
wget -P pretrained_weights/encoder https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/encoder/training_info.yaml

mkdir -p pretrained_weights/transformer/Kathleen2Donald_Trump
wget -P pretrained_weights/transformer/Kathleen2Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/transformer/pretrained_weights.h5
wget -P pretrained_weights/transformer/Kathleen2Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/transformer/training_info.yaml

mkdir -p pretrained_weights/decoder/Donald_Trump
wget -P pretrained_weights/decoder/Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/decoder/weights.h5
wget -P pretrained_weights/decoder/Donald_Trump https://nnabla.org/pretrained-models/nnabla-examples/GANs/reenactgan/decoder/training_info.yaml
