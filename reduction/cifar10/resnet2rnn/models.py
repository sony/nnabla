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


def res_unit_default(x, scope, bn_idx, test):
    # BatchNorm is independent from parameter sharing
    C = x.shape[1]
    with nn.parameter_scope(scope):
        with nn.parameter_scope('conv1'):
            with nn.parameter_scope('bn_{}-a'.format(bn_idx)):
                h = PF.batch_normalization(x, batch_stat=not test)
                h = F.relu(h)
            h = PF.convolution(h, C, (3, 3), pad=(1, 1), with_bias=False)
            with nn.parameter_scope('bn_{}-b'.format(bn_idx)):
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            if not test:
                h = F.dropout(h, 0.25)
        with nn.parameter_scope('conv2'):
            h = PF.convolution(h, C, (3, 3), pad=(1, 1), with_bias=False)
    return x + h


def res_unit_bottleneck(x, scope, bn_idx, test):
    C = x.shape[1]
    with nn.parameter_scope(scope):
        # Conv -> BN -> Relu
        with nn.parameter_scope("conv1"):
            h = PF.convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                               with_bias=False)
            with nn.parameter_scope('bn_{}-a'.format(bn_idx)):
                h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # Conv -> BN -> Relu
        with nn.parameter_scope("conv2"):
            h = PF.convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                               with_bias=False)
            with nn.parameter_scope('bn_{}-b'.format(bn_idx)):
                h = PF.batch_normalization(h, batch_stat=not test)
            h = F.relu(h)
        # Conv -> BN
        with nn.parameter_scope("conv3"):
            h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                               with_bias=False)
            with nn.parameter_scope('bn_{}-c'.format(bn_idx)):
                h = PF.batch_normalization(h, batch_stat=not test)
        # Residual -> Relu
        h = F.relu(h + x)

    return h


def cifar10_resnet2rnn_prediction(image, maps=64, unrolls=[3, 3, 4],
                                  res_unit=res_unit_default,
                                  test=False):
    """
    Construct ResNet 23 with depth-wise convolution.

    References
    ----------
    Qianli Liao and Tomaso Poggio,
    "Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex",
    https://arxiv.org/abs/1604.03640

    """

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

    # ResUnit2RNN
    for i, u in enumerate(unrolls):
        for n in range(u):
            h = res_unit(h, "block{}".format(i), n, test)

    h = F.average_pooling(h, kernel=h.shape[-2:])
    pred = PF.affine(h, ncls)

    return pred
