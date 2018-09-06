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


def cifar10_resnet23_prediction(image, net="teacher", maps=64,
                                test=False):
    """
    Construct ResNet 23
    """
    # Residual Unit
    def res_unit(x, scope_name, dn=False):
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
    with nn.parameter_scope(net):
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


def cifar10_shuffle_prediction(image, maps=64, groups=1, test=False):
    """
    Construct ShuffleNet
    """

    def shuffle(x):
        n, c, h, w = x.shape
        g = groups
        assert c % g == 0

        # N, C, H, W -> N, g, C/g, H, W -> N, C/g, g, H, W -> N, C, H, W
        x = F.reshape(x, [n, g, c // g, h, w])
        x = F.transpose(x, [0, 2, 1, 3, 4])
        x = F.reshape(x, [n, c, h, w])

        return x

    # Shuffle
    def shuffle_unit(x, scope_name, dn=False):
        """
        Figure. 2 (b) and (c) in https://arxiv.org/pdf/1707.01083.pdf
        """

        C = x.shape[1]
        h = x
        with nn.parameter_scope(scope_name):
            with nn.parameter_scope("gconv1"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   group=groups,
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h, True)

            with nn.parameter_scope("shuffle"):  # no meaning but semantics
                h = shuffle(h)

            with nn.parameter_scope("dconv"):
                stride = (2, 2) if dn else (1, 1)
                h = PF.depthwise_convolution(h, kernel=(3, 3), pad=(1, 1),
                                             stride=stride,
                                             with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)

            with nn.parameter_scope("gconv2"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   group=groups,
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)

            s = F.average_pooling(x, (2, 2)) if dn else x
            h = F.concatenate(*[h, s], axis=1) if dn else h + s
            h = F.relu(h)
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

    h = shuffle_unit(h, "conv2", False)    # -> 32x32
    h = shuffle_unit(h, "conv3", True)     # -> 16x16
    h = shuffle_unit(h, "conv4", False)    # -> 16x16
    h = shuffle_unit(h, "conv5", True)     # -> 8x8
    h = shuffle_unit(h, "conv6", False)    # -> 8x8
    h = shuffle_unit(h, "conv7", True)     # -> 4x4
    h = shuffle_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.affine(h, ncls)

    return pred
