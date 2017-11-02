# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from six.moves import range

import os

from args import get_args
from cifar10_data import data_iterator_cifar10
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
import numpy as np


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def cifar10_resnet23_prediction(image, maps=64,
                                test=False):
    """
    Construct Resnet23 as the reference.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                           with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.affine(h, ncls)

    return pred


def cifar10_binary_connect_resnet23_prediction(image, maps=64,
                                               test=False):
    """
    Construct BianryConnect using resnet23. Binary Connect binaries weights.

    References: 
        Courbariaux Matthieu, Bengio Yoshua, David Jean-Pierre, 
        "BinaryConnect: Training Deep Neural Networks with binary weights during propagations", 
        Advances in Neural Information Processing Systems 28 (NIPS 2015)

    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.binary_connect_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.binary_connect_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.binary_connect_convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.binary_connect_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                                          with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.binary_connect_affine(h, ncls)

    return pred


def cifar10_binary_net_resnet23_prediction(image, maps=64,
                                           test=False):
    """
    Construct BianryNet using resnet23. Binary Net binaries weights and activations.


    References: 
        Courbariaux Matthieu, Bengio Yoshua, David Jean-Pierre, 
        "BinaryConnect: Training Deep Neural Networks with binary weights during propagations", 
        Advances in Neural Information Processing Systems 28 (NIPS 2015)

    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> BinaryTanh
            with nn.parameter_scope("conv1"):
                h = PF.binary_connect_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.binary_tanh(h)
            # Conv -> BN -> BinaryTanh
            with nn.parameter_scope("conv2"):
                h = PF.binary_connect_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.binary_tanh(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.binary_connect_convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> BinaryTanh
            h = F.binary_tanh(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Binary_tanh
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.binary_connect_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                                          with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.binary_tanh(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.binary_connect_affine(h, ncls)

    return pred


def cifar10_binary_weight_resnet23_prediction(image, maps=64,
                                              test=False):
    """
    Construct BianryWeight using resnet23. Binary Weight binaries weights, but 
    use the approximate coefficients to alleviate the binary quantization.

    References:
        Rastegari Mohammad, Ordonez Vicente, Redmon Joseph, and Farhadi Ali, 
        "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks", 
        arXiv:1603.05279 

    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.binary_weight_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                                 with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.binary_weight_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                                 with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.binary_weight_convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                                 with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.binary_weight_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                                         with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.binary_weight_affine(h, ncls)

    return pred


def cifar10_fp_connect_resnet23_prediction(image, maps=64,
                                           test=False):
    """
    Construct Fixed-Point Connect using resnet23. Fixed-Point Connect quantizes weights using
    FixedPointQuantize function. 
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.fixed_point_quantized_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.fixed_point_quantized_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.fixed_point_quantized_convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.fixed_point_quantized_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                                                 with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.fixed_point_quantized_affine(h, ncls)

    return pred


def cifar10_fp_net_resnet23_prediction(image, maps=64,
                                       test=False):
    """
    Construct Fixed-Point Net using resnet23. Fixed-Point Net quantizes weights 
    and activations using FixedPointQuantize function.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> FixedPointQuantize -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.fixed_point_quantized_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.fixed_point_quantize(h)
                h = F.relu(h)
            # Conv -> BN -> FixedPointQuantize -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.fixed_point_quantized_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.fixed_point_quantize(h)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.fixed_point_quantized_convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> FixedPointQuantize -> Relu
            h = F.fixed_point_quantize(h)
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> FixedPointQuantize
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.fixed_point_quantized_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                                                 with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.fixed_point_quantize(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.fixed_point_quantized_affine(h, ncls)

    return pred


def cifar10_pow2_connect_resnet23_prediction(image, maps=64,
                                             test=False):
    """
    Construct Pow2 Connect using resnet23. Pow2 Connect quantizes weights 
    using Pow2Quantize function.

    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.pow2_quantized_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.pow2_quantized_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.pow2_quantized_convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Relu
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.pow2_quantized_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                                          with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.pow2_quantized_affine(h, ncls)

    return pred


def cifar10_pow2_net_resnet23_prediction(image, maps=64,
                                         test=False):
    """
    Construct Pow2 Net using resnet23. Pow2 Net quantizes weights and 
    activations using Pow2Quantize function.

    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Pow2Quantize -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.pow2_quantized_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.pow2_quantize(h)
                h = F.relu(h)
            # Conv -> BN -> Pow2Quantize -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.pow2_quantized_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.pow2_quantize(h)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.pow2_quantized_convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Pow2Quantize -> Relu
            h = F.pow2_quantize(h)
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN -> Pow2Quantize
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.pow2_quantized_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
                                          with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.pow2_quantize(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.pow2_quantized_affine(h, ncls)

    return pred


def train():
    args = get_args()

    # Get context.
    from nnabla.contrib.context import extension_context
    extension_module = args.context
    if args.context is None:
        extension_module = 'cpu'
    logger.info("Running in %s" % extension_module)
    ctx = extension_context(extension_module, device_id=args.device_id)
    nn.set_default_context(ctx)

    # Create CNN network for both training and testing.
    if args.net == "resnet23":
        model_prediction = cifar10_resnet23_prediction
    elif args.net == 'bincon_resnet23':
        model_prediction = cifar10_binary_connect_resnet23_prediction
    elif args.net == 'binnet_resnet23':
        model_prediction = cifar10_binary_net_resnet23_prediction
    elif args.net == 'bwn_resnet23':
        model_prediction = cifar10_binary_weight_resnet23_prediction
    elif args.net == 'fpcon_resnet23':
        model_prediction = cifar10_fp_connect_resnet23_prediction
    elif args.net == 'fpnet_resnet23':
        model_prediction = cifar10_fp_net_resnet23_prediction
    elif args.net == 'pow2con_resnet23':
        model_prediction = cifar10_pow2_connect_resnet23_prediction
    elif args.net == 'pow2net_resnet23':
        model_prediction = cifar10_pow2_net_resnet23_prediction

    # TRAIN
    maps = 64
    c = 3
    h = w = 32
    n_train = 50000
    n_valid = 10000

    # Create input variables.
    image = nn.Variable([args.batch_size, c, h, w])
    label = nn.Variable([args.batch_size, 1])
    # Create model_prediction graph.
    pred = model_prediction(image, maps=maps, test=False)
    pred.persistent = True
    # Create loss function.
    loss = F.mean(F.softmax_cross_entropy(pred, label))

    # TEST
    # Create input variables.
    vimage = nn.Variable([args.batch_size, c, h, w])
    vlabel = nn.Variable([args.batch_size, 1])
    # Create predition graph.
    vpred = model_prediction(vimage, maps=maps, test=True)

    # Create Solver.
    solver = S.Adam(args.learning_rate)
    solver.set_parameters(nn.get_parameters())

    # Create monitor.
    from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_err = MonitorSeries("Training error", monitor, interval=10)
    monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
    monitor_verr = MonitorSeries("Test error", monitor, interval=10)

    # Initialize DataIterator
    best_ve = 1.0
    ve = 1.0
    tdata = data_iterator_cifar10(args.batch_size, True)
    vdata = data_iterator_cifar10(args.batch_size, False)
    # Training loop
    for i in range(args.max_iter):
        if i % args.val_interval == 0:
            # Validation
            ve = 0.0
            for j in range(int(n_valid / args.batch_size)):
                vimage.d, vlabel.d = vdata.next()
                vpred.forward(clear_buffer=True)
                ve += categorical_error(vpred.d, vlabel.d)
            ve /= int(n_valid / args.batch_size)
            monitor_verr.add(i, ve)
        if ve < best_ve:
            nn.save_parameters(os.path.join(
                args.model_save_path, 'params_%06d.h5' % i))
            best_ve = ve

        # Training forward
        image.d, label.d = tdata.next()
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(args.weight_decay)
        solver.update()
        e = categorical_error(pred.d, label.d)
        monitor_loss.add(i, loss.d.copy())
        monitor_err.add(i, e)
        monitor_time.add(i)

    ve = 0.0
    for j in range(int(n_valid / args.batch_size)):
        vimage.d, vlabel.d = vdata.next()
        vpred.forward(clear_buffer=True)
        ve += categorical_error(vpred.d, vlabel.d)
    ve /= int(n_valid / args.batch_size)
    monitor_verr.add(i, ve)

    parameter_file = os.path.join(
        args.model_save_path, 'params_{:06}.h5'.format(args.max_iter))
    nn.save_parameters(parameter_file)


if __name__ == '__main__':
    train()
