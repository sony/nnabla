# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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


import os
import time
from args import get_args
import nnabla as nn
import nnabla.communicators as C
from nnabla.ext_utils import get_extension_context
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def resnet18_prediction(image, test=False, ncls=10, nmaps=64, act=F.relu):
    """
    Construct ResNet 18
    """
    # Residual Unit
    def res_unit(x, nmap_out, scope_name, stride=1):
        nmap_in = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, nmap_out, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False, stride=(stride, stride))
                h = PF.batch_normalization(h, batch_stat=not test)
                h = act(h)
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, nmap_out, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Conv -> BN
            if nmap_in != nmap_out:
                with nn.parameter_scope("conv3"):
                    x2 = PF.convolution(x, nmap_out, kernel=(1, 1), pad=(0, 0),
                                        with_bias=False, stride=(stride, stride))
                    x2 = PF.batch_normalization(x2, batch_stat=not test)
            else:
                x2 = x
            # Residual -> Nonlinear
            h = act(F.add2(h, x2))
            return h
    # Conv -> BN -> Nonlinear
    with nn.parameter_scope("conv1"):
        h = PF.convolution(image, nmaps, kernel=(3, 3),
                           pad=(1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)

    h = res_unit(h, nmaps, "conv2-1", 1)    # -> 32x32
    h = res_unit(h, nmaps, "conv2-2", 1)    # -> 32x32
    h = res_unit(h, nmaps*2, "conv3-1", 2)  # -> 16x16
    h = res_unit(h, nmaps*2, "conv3-2", 1)  # -> 16x16
    h = res_unit(h, nmaps*4, "conv4-1", 2)  # -> 8x8
    h = res_unit(h, nmaps*4, "conv4-2", 1)  # -> 8x8
    h = res_unit(h, nmaps*8, "conv5-1", 2)  # -> 4x4
    h = res_unit(h, nmaps*8, "conv5-2", 1)  # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    h = PF.affine(h, 1000, name="bottleneck")  # -> 1x1000
    h = act(h)
    pred = PF.affine(h, ncls)

    return pred


def resnet34_prediction(image, test=False, ncls=10, nmaps=64, act=F.relu):
    """
    Construct ResNet 34
    """
    # Residual Unit
    def res_unit(x, nmap_out, scope_name, stride=1):
        nmap_in = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, nmap_out, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False, stride=(stride, stride))
                h = PF.batch_normalization(h, batch_stat=not test)
                h = act(h)
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, nmap_out, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
            # Conv -> BN
            if nmap_in != nmap_out:
                with nn.parameter_scope("conv3"):
                    x2 = PF.convolution(x, nmap_out, kernel=(1, 1), pad=(0, 0),
                                        with_bias=False, stride=(stride, stride))
                    x2 = PF.batch_normalization(x2, batch_stat=not test)
            else:
                x2 = x
            # Residual -> Nonlinear
            h = act(F.add2(h, x2))
            return h
    # Conv -> BN -> Nonlinear
    with nn.parameter_scope("conv1"):
        h = PF.convolution(image, nmaps, kernel=(3, 3),
                           pad=(1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = act(h)

    h = res_unit(h, nmaps, "conv2-1", 1)    # -> 32x32
    h = res_unit(h, nmaps, "conv2-2", 1)    # -> 32x32
    h = res_unit(h, nmaps, "conv2-3", 1)    # -> 32x32
    h = res_unit(h, nmaps*2, "conv3-1", 2)  # -> 16x16
    h = res_unit(h, nmaps*2, "conv3-2", 1)  # -> 16x16
    h = res_unit(h, nmaps*2, "conv3-3", 1)  # -> 16x16
    h = res_unit(h, nmaps*2, "conv3-4", 1)  # -> 16x16
    h = res_unit(h, nmaps*4, "conv4-1", 2)  # -> 8x8
    h = res_unit(h, nmaps*4, "conv4-2", 1)  # -> 8x8
    h = res_unit(h, nmaps*4, "conv4-3", 1)  # -> 8x8
    h = res_unit(h, nmaps*4, "conv4-4", 1)  # -> 8x8
    h = res_unit(h, nmaps*4, "conv4-5", 1)  # -> 8x8
    h = res_unit(h, nmaps*4, "conv4-6", 1)  # -> 8x8
    h = res_unit(h, nmaps*8, "conv5-1", 2)  # -> 4x4
    h = res_unit(h, nmaps*8, "conv5-2", 1)  # -> 4x4
    h = res_unit(h, nmaps*8, "conv5-3", 1)  # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    h = PF.affine(h, 1000, name="bottleneck")  # -> 1x1000
    h = act(h)
    pred = PF.affine(h, ncls)

    return pred
