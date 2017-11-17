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

from six.moves import range

import os

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save


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
    Construct Resnet23 as reference.
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
    Construct BianryConnect using resnet23.
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
    Construct BianryNet using resnet23.
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
    Construct BianryWeight using resnet23.
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


def cifar10_fp_connect_resnet23_prediction(image, maps=64, n=8, delta=2**-4,
                                           test=False):
    """
    Construct FixedPointConnect using resnet23.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.fixed_point_quantized_convolution(x, C / 2,
                                                         kernel=(1, 1), pad=(0, 0),
                                                         n_w=n, delta_w=delta,
                                                         n_b=n, delta_b=delta,
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.fixed_point_quantized_convolution(h, C / 2,
                                                         kernel=(3, 3), pad=(1, 1),
                                                         n_w=n, delta_w=delta,
                                                         n_b=n, delta_b=delta,
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.fixed_point_quantized_convolution(h, C,
                                                         kernel=(1, 1), pad=(0, 0),
                                                         n_w=n, delta_w=delta,
                                                         n_b=n, delta_b=delta,
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
        h = PF.fixed_point_quantized_convolution(image, maps,
                                                 kernel=(3, 3), pad=(1, 1),
                                                 n_w=n, delta_w=delta,
                                                 n_b=n, delta_b=delta,
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
    pred = PF.fixed_point_quantized_affine(h, ncls,
                                           n_w=n, delta_w=delta,
                                           n_b=n, delta_b=delta)

    return pred


def cifar10_fp_net_resnet23_prediction(image, maps=64, n=8, delta=2**-4,
                                       test=False):
    """
    Construct FixedPointNet using resnet23.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> FixedPointQuantize -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.fixed_point_quantized_convolution(x, C / 2,
                                                         kernel=(1, 1), pad=(0, 0),
                                                         n_w=n, delta_w=delta,
                                                         n_b=n, delta_b=delta,
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.fixed_point_quantize(h, n=n, delta=delta)
                h = F.relu(h)
            # Conv -> BN -> FixedPointQuantize -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.fixed_point_quantized_convolution(h, C / 2,
                                                         kernel=(3, 3), pad=(1, 1),
                                                         n_w=n, delta_w=delta,
                                                         n_b=n, delta_b=delta,
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.fixed_point_quantize(h, n=n, delta=delta)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.fixed_point_quantized_convolution(h, C,
                                                         kernel=(1, 1), pad=(0, 0),
                                                         n_w=n, delta_w=delta,
                                                         n_b=n, delta_b=delta,
                                                         with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> FixedPointQuantize -> Relu
            h = F.fixed_point_quantize(h, n=n, delta=delta)
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
        h = PF.fixed_point_quantized_convolution(image, maps,
                                                 kernel=(3, 3), pad=(1, 1),
                                                 n_w=n, delta_w=delta,
                                                 n_b=n, delta_b=delta,
                                                 with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.fixed_point_quantize(h, n=n, delta=delta)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.fixed_point_quantized_affine(h, ncls,
                                           n_w=n, delta_w=delta,
                                           n_b=n, delta_b=delta)

    return pred


def cifar10_pow2_connect_resnet23_prediction(image, maps=64, n=8, m=1,
                                             test=False):
    """
    Construct Pow2Connect using resnet23.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.pow2_quantized_convolution(x, C / 2,
                                                  kernel=(1, 1), pad=(0, 0),
                                                  n_w=n, m_w=m,
                                                  n_b=n, m_b=m,
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.pow2_quantized_convolution(h, C / 2,
                                                  kernel=(3, 3), pad=(1, 1),
                                                  n_w=n, m_w=m,
                                                  n_b=n, m_b=m,
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.pow2_quantized_convolution(h, C,
                                                  kernel=(1, 1), pad=(0, 0),
                                                  n_w=n, m_w=m,
                                                  n_b=n, m_b=m,
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
        h = PF.pow2_quantized_convolution(image, maps,
                                          kernel=(3, 3), pad=(1, 1),
                                          n_w=n, m_w=m,
                                          n_b=n, m_b=m,
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
    pred = PF.pow2_quantized_affine(h, ncls,
                                    n_w=n, m_w=m,
                                    n_b=n, m_b=m)

    return pred


def cifar10_pow2_net_resnet23_prediction(image, maps=64, n=8, m=1,
                                         test=False):
    """
    Construct Pow2Net using resnet23.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Pow2Quantize -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.pow2_quantized_convolution(x, C / 2,
                                                  kernel=(1, 1), pad=(0, 0),
                                                  n_w=n, m_w=m,
                                                  n_b=n, m_b=m,
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.pow2_quantize(h, n=n, m=m)
                h = F.relu(h)
            # Conv -> BN -> Pow2Quantize -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.pow2_quantized_convolution(h, C / 2,
                                                  kernel=(3, 3), pad=(1, 1),
                                                  n_w=n, m_w=m,
                                                  n_b=n, m_b=m,
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.pow2_quantize(h, n=n, m=m)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.pow2_quantized_convolution(h, C,
                                                  kernel=(1, 1), pad=(0, 0),
                                                  n_w=n, m_w=m,
                                                  n_b=n, m_b=m,
                                                  with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Pow2Quantize -> Relu
            h = F.pow2_quantize(h, n=n, m=m)
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
        h = PF.pow2_quantized_convolution(image, maps,
                                          kernel=(3, 3), pad=(1, 1),
                                          n_w=n, m_w=m,
                                          n_b=n, m_b=m,
                                          with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.pow2_quantize(h, n=n, m=m)
        h = F.relu(h)

    h = res_unit(h, "conv2", False)    # -> 32x32
    h = res_unit(h, "conv3", True)     # -> 16x16
    h = res_unit(h, "conv4", False)    # -> 16x16
    h = res_unit(h, "conv5", True)     # -> 8x8
    h = res_unit(h, "conv6", False)    # -> 8x8
    h = res_unit(h, "conv7", True)     # -> 4x4
    h = res_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.pow2_quantized_affine(h, ncls,
                                    n_w=n, m_w=m,
                                    n_b=n, m_b=m)

    return pred


def cifar10_inq_resnet23_prediction(image, maps=64, num_bits=4,
                                    inq_iterations=(
                                        5000, 6000, 7000, 8000, 9000),
                                    selection_algorithm='largest_abs',
                                    test=False):
    """
    Construct INQ Network using resnet23.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = PF.inq_convolution(x, C / 2,
                                       kernel=(1, 1), pad=(0, 0),
                                       inq_iterations=inq_iterations,
                                       selection_algorithm=selection_algorithm,
                                       with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = PF.inq_convolution(h, C / 2,
                                       kernel=(3, 3), pad=(1, 1),
                                       inq_iterations=inq_iterations,
                                       selection_algorithm=selection_algorithm,
                                       with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.inq_convolution(h, C,
                                       kernel=(1, 1), pad=(0, 0),
                                       inq_iterations=inq_iterations,
                                       selection_algorithm=selection_algorithm,
                                       with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Residual -> Relu
            h = F.relu(h + x)

            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))

        return h

    ncls = 10

    # Conv -> BN
    with nn.parameter_scope("conv1"):
        # Preprocess
        image /= 255.0
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.inq_convolution(image, maps,
                               kernel=(3, 3), pad=(1, 1),
                               inq_iterations=inq_iterations,
                               selection_algorithm=selection_algorithm,
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
    pred = PF.inq_affine(h, ncls,
                         inq_iterations=inq_iterations,
                         selection_algorithm=selection_algorithm)
    return pred
