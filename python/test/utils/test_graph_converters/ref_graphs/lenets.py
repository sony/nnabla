# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from nnabla.parameter import get_parameter_or_create
from .helper import get_channel_axes

# LeNet


def lenet(image, test=False, w_bias=False):
    h = PF.convolution(image, 16, (5, 5), (1, 1),
                       with_bias=False, name='conv1')
    h = PF.batch_normalization(h, batch_stat=not test, name='conv1-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name='conv2')
    h = PF.batch_normalization(h, batch_stat=not test, name='conv2-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=False, name='fc1')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# Channel last LeNet
def cl_lenet(image, test=False):
    h = PF.convolution(image, 16, (5, 5), (1, 1),
                       channel_last=True, with_bias=False, name='conv1')
    h = PF.batch_normalization(h, batch_stat=not test, name='conv1-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1),
                       channel_last=True, with_bias=True, name='conv2')
    h = PF.batch_normalization(h, batch_stat=not test, name='conv2-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=False, name='fc1')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# BN LeNet
def bn_lenet(image, test=False, channel_last=False, w_bias=False):
    axes = get_channel_axes(channel_last)
    h = PF.convolution(image, 16, (5, 5), (1, 1), with_bias=w_bias,
                       channel_last=channel_last, name='conv1')
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='conv1-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True,
                       channel_last=channel_last, name='conv2')
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='conv2-bn')
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=True, name='fc1')
    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='fc1-bn')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# Batch Normalization Small LeNet
def bn_folding_lenet(image, test=False, channel_last=False, name="bn-folding-graph-ref"):
    h = PF.convolution(image, 16, (5, 5), (1, 1), with_bias=True,
                       channel_last=channel_last, name='conv1')
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True,
                       channel_last=channel_last, name='conv2')
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=True, name='fc1')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# LeNet with batch_stat False
def bsf_lenet(image, test=False, w_bias=False):
    h = PF.convolution(image, 16, (5, 5), (1, 1),
                       with_bias=False, name='conv1')
    h = PF.batch_normalization(h, batch_stat=False, name='conv1-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name='conv2')
    h = PF.batch_normalization(h, batch_stat=False, name='conv2-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=False, name='fc1')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# BN LeNet Opposite
def bn_opp_lenet(image, test=False, channel_last=False, w_bias=False):
    axes = get_channel_axes(channel_last)
    h = PF.batch_normalization(
        image, axes=axes, batch_stat=not test, name='conv1-bn')
    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=w_bias,
                       channel_last=channel_last, name='conv1')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='conv2-bn')
    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True,
                       channel_last=channel_last, name='conv2')
    h = F.max_pooling(h, (2, 2), channel_last=channel_last)
    h = F.relu(h)

    h = PF.batch_normalization(
        h, axes=axes, batch_stat=not test, name='fc1-bn')
    h = PF.affine(h, 10, with_bias=True, name='fc1')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# Identity LeNet
def id_lenet(image, test=False):
    h = PF.convolution(image, 16, (5, 5), (1, 1),
                       with_bias=False, name='id-conv1')
    h = PF.batch_normalization(h, batch_stat=not test, name='id-conv1-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name='id-conv2')
    h = PF.batch_normalization(h, batch_stat=not test, name='id-conv2-bn')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=False, name='id-fc1')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='id-fc2')
    return pred
