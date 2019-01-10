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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from nnabla.logger import logger
from nnabla.config import nnabla_config
from nnabla.utils.data_source_loader import load_image
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.utils.data_iterator import data_iterator_simple

from nnabla.experimental import graph_converters as GC

from .helper import create_scale_bias


# Small ResNet
def resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
    return F.relu(h + x)


def small_resnet(image, test=False):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(
        1, 1), name="first-conv", with_bias=False)
    h = PF.batch_normalization(h, name="first-bn", batch_stat=not test)
    h = F.relu(h)
    h = resblock(h, maps=16, test=test, name="cb1")
    h = resblock(h, maps=16, test=test, name="cb2")
    h = resblock(h, maps=16, test=test, name="cb3")
    h = resblock(h, maps=16, test=test, name="cb4")
    pred = PF.affine(h, 10, name='fc')
    return pred


# Small Fixed-point-weight ResNet
def fpq_weight_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False,
                        sign=True, n=8, delta=2e-4,
                        name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.fixed_point_quantized_convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                                                 with_bias=False,
                                                 sign_w=sign, n_w=n, delta_w=delta,
                                                 sign_b=sign, n_b=n, delta_b=delta)
        h = PF.batch_normalization(h, batch_stat=not test)
    return F.relu(h + x)


def fpq_weight_small_resnet(image, test=False, sign=True, n=8, delta=2e-4,
                            name="fixed-point-graph-ref"):
    with nn.parameter_scope(name):
        h = image
        h /= 255.0
        h = PF.fixed_point_quantized_convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                                                 sign_w=sign, n_w=n, delta_w=delta,
                                                 sign_b=sign, n_b=n, delta_b=delta,
                                                 with_bias=False,
                                                 name="first-conv")
        h = PF.batch_normalization(h, name="first-bn", batch_stat=not test)
        h = F.relu(h)
        h = fpq_weight_resblock(h, maps=16, test=test,
                                sign=sign, n=n, delta=delta, name="cb1")
        h = fpq_weight_resblock(h, maps=16, test=test,
                                sign=sign, n=n, delta=delta, name="cb2")
        h = fpq_weight_resblock(h, maps=16, test=test,
                                sign=sign, n=n, delta=delta, name="cb3")
        h = fpq_weight_resblock(h, maps=16, test=test,
                                sign=sign, n=n, delta=delta, name="cb4")
        pred = PF.fixed_point_quantized_affine(h, 10,
                                               sign_w=sign, n_w=n, delta_w=delta,
                                               sign_b=sign, n_b=n, delta_b=delta,
                                               name='fc')
    return pred


# Small Fixed-point-activation ResNet
def fpq_relu_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False,
                      sign=False, n=8, delta=2e-4,
                      name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
    return F.fixed_point_quantize(h + x, n=n, sign=sign, delta=delta)


def fpq_relu_small_resnet(image, test=False,
                          sign=False, n=8, delta=2e-4,
                          name="fixed-point-graph-ref"):
    with nn.parameter_scope(name):
        h = image
        h /= 255.0
        h = PF.convolution(h, 16, kernel=(3, 3), pad=(
            1, 1), name="first-conv", with_bias=False)
        h = PF.batch_normalization(h, name="first-bn", batch_stat=not test)
        h = F.fixed_point_quantize(h, sign=sign, n=n, delta=delta)
        h = fpq_relu_resblock(h, maps=16, test=test,
                              sign=sign, n=n, delta=delta, name="cb1")
        h = fpq_relu_resblock(h, maps=16, test=test,
                              sign=sign, n=n, delta=delta, name="cb2")
        h = fpq_relu_resblock(h, maps=16, test=test,
                              sign=sign, n=n, delta=delta, name="cb3")
        h = fpq_relu_resblock(h, maps=16, test=test,
                              sign=sign, n=n, delta=delta, name="cb4")
        pred = PF.affine(h, 10, name='fc')
    return pred


# Small Fixed-point ResNet
def fpq_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False,
                 sign=True, n=8, delta=2e-4,
                 name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.fixed_point_quantized_convolution(h, maps, kernel=kernel, pad=pad, stride=stride,
                                                 with_bias=False,
                                                 sign_w=sign, n_w=n, delta_w=delta,
                                                 sign_b=sign, n_b=n, delta_b=delta)
        h = PF.batch_normalization(h, batch_stat=not test)
    return F.fixed_point_quantize(h + x, n=n, sign=sign, delta=delta)


def fpq_small_resnet(image, test=False, sign=True, n=8, delta=2e-4,
                     name="fixed-point-graph-ref"):
    with nn.parameter_scope(name):
        h = image
        h /= 255.0
        h = PF.fixed_point_quantized_convolution(h, 16, kernel=(3, 3), pad=(1, 1),
                                                 sign_w=sign, n_w=n, delta_w=delta,
                                                 sign_b=sign, n_b=n, delta_b=delta,
                                                 with_bias=False,
                                                 name="first-conv")
        h = PF.batch_normalization(h, name="first-bn", batch_stat=not test)
        h = F.fixed_point_quantize(h, n=n, sign=sign, delta=delta)
        h = fpq_resblock(h, maps=16, test=test, sign=sign,
                         n=n, delta=delta, name="cb1")
        h = fpq_resblock(h, maps=16, test=test, sign=sign,
                         n=n, delta=delta, name="cb2")
        h = fpq_resblock(h, maps=16, test=test, sign=sign,
                         n=n, delta=delta, name="cb3")
        h = fpq_resblock(h, maps=16, test=test, sign=sign,
                         n=n, delta=delta, name="cb4")
        pred = PF.fixed_point_quantized_affine(h, 10,
                                               sign_w=sign, n_w=n, delta_w=delta,
                                               sign_b=sign, n_b=n, delta_b=delta,
                                               name='fc')
    return pred

# BatchNormalization Linear Small ResNet


def bn_linear_resblock(x, i, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=False)
        a, b = create_scale_bias(i, h.shape[1])
        h = a * h + b
    return F.relu(h + x)


def bn_linear_small_resnet(image,
                           name="bn-linear-graph-ref"):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(
        1, 1), name="first-conv", with_bias=False)
    a, b = create_scale_bias(1, h.shape[1])
    h = a * h + b
    h = F.relu(h)
    h = bn_linear_resblock(h, 2, maps=16, name="cb1")
    h = bn_linear_resblock(h, 3, maps=16, name="cb2")
    h = bn_linear_resblock(h, 4, maps=16, name="cb3")
    h = bn_linear_resblock(h, 5, maps=16, name="cb4")
    pred = PF.affine(h, 10, name='fc')
    return pred


# Batch Normalization Folded Small ResNet
def bn_folded_resblock(x, maps, kernel=(3, 3), pad=(1, 1), stride=(1, 1), test=False, name="convblock"):
    h = x
    with nn.parameter_scope(name):
        h = PF.convolution(h, maps, kernel=kernel, pad=pad,
                           stride=stride, with_bias=True)
    return F.relu(h + x)


def bn_folded_small_resnet(image, test=False,
                           name="bn-folded-graph-ref"):
    h = image
    h /= 255.0
    h = PF.convolution(h, 16, kernel=(3, 3), pad=(
        1, 1), name="first-conv", with_bias=True)
    h = F.relu(h)
    h = bn_folded_resblock(h, maps=16, test=test, name="cb1")
    h = bn_folded_resblock(h, maps=16, test=test, name="cb2")
    h = bn_folded_resblock(h, maps=16, test=test, name="cb3")
    h = bn_folded_resblock(h, maps=16, test=test, name="cb4")
    pred = PF.affine(h, 10, name='fc')
    return pred
