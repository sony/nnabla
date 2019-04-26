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


import os
import time
from args import get_args
from cifar10_data import data_iterator_cifar10
from cifar100_data import data_iterator_cifar100
import nnabla as nn
import nnabla.communicators as C
from nnabla.ext_utils import get_extension_context
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np


def categorical_error(pred, label):
    # TODO: Use F.top_n_error
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def batch_normalization(h, test=False, comm=None, group="world"):
    if comm is None:
        h = PF.batch_normalization(h, batch_stat=not test)
    else:
        h = PF.sync_batch_normalization(
            h, comm=comm, group=group, batch_stat=not test)
    return h


def resnet23_prediction(image, test=False, rng=None, ncls=10, nmaps=64, act=F.relu, comm=None, group="world"):
    """
    Construct ResNet 23
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv1"):
                h = PF.convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = batch_normalization(h, test=test, comm=comm, group=group)
                h = act(h)
            # Conv -> BN -> Nonlinear
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                   with_bias=False)
                h = batch_normalization(h, test=test, comm=comm, group=group)
                h = act(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)
                h = batch_normalization(h, test=test, comm=comm, group=group)
            # Residual -> Nonlinear
            h = act(F.add2(h, x, inplace=False))
            # Maxpooling
            if dn:
                h = F.max_pooling(h, kernel=(2, 2), stride=(2, 2))
            return h
    # Conv -> BN -> Nonlinear
    with nn.parameter_scope("conv1"):
        # Preprocess
        if not test:
            image = F.image_augmentation(image, contrast=1.0,
                                         angle=0.25,
                                         flip_lr=True)
            image.need_grad = False
        h = PF.convolution(image, nmaps, kernel=(3, 3),
                           pad=(1, 1), with_bias=False)
        h = batch_normalization(h, test=test, comm=comm, group=group)
        h = act(h)

    h = res_unit(h, "conv2", rng, False)    # -> 32x32
    h = res_unit(h, "conv3", rng, True)     # -> 16x16
    h = res_unit(h, "conv4", rng, False)    # -> 16x16
    h = res_unit(h, "conv5", rng, True)     # -> 8x8
    h = res_unit(h, "conv6", rng, False)    # -> 8x8
    h = res_unit(h, "conv7", rng, True)     # -> 4x4
    h = res_unit(h, "conv8", rng, False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.affine(h, ncls, rng=rng)

    return pred


def loss_function(pred, label):
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss
