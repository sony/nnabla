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

import os
import pytest

import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from nnabla.parameter import get_parameter_or_create

from .helper import create_scale_bias

# LeNet


def lenet(image, test=False):
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
    h = PF.batch_normalization(h, batch_stat=not test, name='fc1-bn')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# Small Fixed-point-weight LeNet
def fpq_weight_lenet(image, test=False, n=8, delta=2e-4, name="fixed-point-weight-graph-ref"):
    with nn.parameter_scope(name):
        h = PF.fixed_point_quantized_convolution(image, 16, (5, 5), (1, 1),
                                                 with_bias=False, n_w=n, delta_w=delta,
                                                 name='conv1')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv1-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.relu(h)

        h = PF.fixed_point_quantized_convolution(h, 16, (5, 5), (1, 1),
                                                 with_bias=True, n_w=n, delta_w=delta,
                                                 name='conv2')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv2-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.relu(h)

        h = PF.fixed_point_quantized_affine(
            h, 10, with_bias=False, n_w=n, delta_w=delta, name='fc1')
        h = PF.batch_normalization(h, batch_stat=not test, name='fc1-bn')
        h = F.relu(h)

        pred = PF.fixed_point_quantized_affine(
            h, 10, with_bias=True, n_w=n, delta_w=delta, name='fc2')
    return pred

# Small Fixed-point-activation LeNet


def fpq_relu_lenet(image, test=False, n=8, delta=2e-4, name="fixed-point-relu-graph-ref"):
    with nn.parameter_scope(name):
        h = PF.convolution(image, 16, (5, 5), (1, 1),
                           with_bias=False, name='conv1')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv1-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.fixed_point_quantize(h, n=n, delta=delta, sign=False)

        h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name='conv2')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv2-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.fixed_point_quantize(h, n=n, delta=delta, sign=False)

        h = PF.affine(h, 10, with_bias=False, name='fc1')
        h = PF.batch_normalization(h, batch_stat=not test, name='fc1-bn')
        h = F.fixed_point_quantize(h, n=n, delta=delta, sign=False)

        pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# Small Fixed-point LeNet
def fpq_lenet(image, test=False, n=8, delta=2e-4, name="fixed-point-graph-ref"):
    with nn.parameter_scope(name):
        h = PF.fixed_point_quantized_convolution(image, 16, (5, 5), (1, 1),
                                                 with_bias=False, delta_w=delta,
                                                 name='conv1')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv1-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.fixed_point_quantize(h, n=n, delta=delta, sign=False)

        h = PF.fixed_point_quantized_convolution(h, 16, (5, 5), (1, 1),
                                                 with_bias=True, delta_w=delta,
                                                 name='conv2')
        h = PF.batch_normalization(h, batch_stat=not test, name='conv2-bn')
        h = F.max_pooling(h, (2, 2))
        h = F.fixed_point_quantize(h, n=n, delta=delta, sign=False)

        h = PF.fixed_point_quantized_affine(
            h, 10, with_bias=False, delta_w=delta, name='fc1')
        h = PF.batch_normalization(h, batch_stat=not test, name='fc1-bn')
        h = F.fixed_point_quantize(h, n=n, delta=delta, sign=False)

        pred = PF.fixed_point_quantized_affine(
            h, 10, with_bias=True, delta_w=delta, name='fc2')
    return pred


# BatchNormalization Linear Small LeNet
def bn_linear_lenet(image, test=False, name="bn-linear-graph-ref"):
    with nn.parameter_scope(name):
        h = PF.convolution(image, 16, (5, 5), (1, 1),
                           with_bias=False, name='conv1')
        a, b = create_scale_bias(1, h.shape[1])
        h = a * h + b
        h = F.max_pooling(h, (2, 2))
        h = F.relu(h)

        h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name='conv2')
        a, b = create_scale_bias(2, h.shape[1])
        h = a * h + b
        h = F.max_pooling(h, (2, 2))
        h = F.relu(h)

        h = PF.affine(h, 10, with_bias=False, name='fc1')
        a, b = create_scale_bias(4, h.shape[1], 2)
        h = a * h + b
        h = F.relu(h)

        pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred


# Batch Normalization Small LeNet
def bn_folded_lenet(image, test=False, name="bn-folded-graph-ref"):
    h = PF.convolution(image, 16, (5, 5), (1, 1), with_bias=True, name='conv1')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.convolution(h, 16, (5, 5), (1, 1), with_bias=True, name='conv2')
    h = F.max_pooling(h, (2, 2))
    h = F.relu(h)

    h = PF.affine(h, 10, with_bias=True, name='fc1')
    h = F.relu(h)

    pred = PF.affine(h, 10, with_bias=True, name='fc2')
    return pred
