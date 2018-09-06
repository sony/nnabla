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


def cifar10_shift_prediction(image, maps=64,
                             test=False, p=0, module="sc2"):
    """
    Construct ShifNet
    """

    # Shift
    def shift(x, ksize=3):
        maps = x.shape[1]
        cpg = maps // (ksize ** 2)

        x_pad = F.pad(x, (1, 1, 1, 1))
        b, c, h, w = x_pad.shape
        xs = []

        # Bottom shift
        i = 0
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, :h-2, 1:w-1]]

        # Top shift
        i = 1
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, 2:, 1:w-1]]

        # Right shift
        i = 2
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, 1:h-1, :w-2]]

        # Left shift
        i = 3
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, 1:h-1, 2:]]

        # Bottom Right shift
        i = 4
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, :h-2, :w-2]]

        # Bottom Left shift
        i = 5
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, :h-2, 2:]]

        # Top Right shift
        i = 6
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, 2:, :w-2]]

        # Top Left shift
        i = 7
        xs += [x_pad[:, i * cpg: (i + 1) * cpg, 2:, 2:]]

        i = 8
        xs += [x_pad[:, i * cpg:, 1:h-1, 1:w-1]]

        h = F.concatenate(*xs, axis=1)
        return h

    # Shift Units

    def sc2(x, scope_name, dn=False):
        C = x.shape[1]
        h = x
        with nn.parameter_scope(scope_name):

            with nn.parameter_scope("shift1"):  # no meaning but semantics
                h = shift(h)

            with nn.parameter_scope("conv1"):
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h, True)
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)

            with nn.parameter_scope("shift2"):  # no meaning but semantics
                h = shift(h)

            with nn.parameter_scope("conv2"):
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h, True)
                stride = (2, 2) if dn else (1, 1)
                if p > 0:
                    h = F.dropout(h, p=0.5) if not test else h
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   stride=stride,
                                   with_bias=False)
        s = F.average_pooling(x, (2, 2)) if dn else x
        return h + s

    def csc(x, scope_name, dn=False):
        C = x.shape[1]
        h = x
        with nn.parameter_scope(scope_name):

            with nn.parameter_scope("conv1"):
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h, True)
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   with_bias=False)

            with nn.parameter_scope("shift"):  # no meaning but semantics
                h = shift(h)

            with nn.parameter_scope("conv2"):
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h, True)
                stride = (2, 2) if dn else (1, 1)
                if p > 0:
                    h = F.dropout(h, p=0.5) if not test else h
                h = PF.convolution(h, C, kernel=(1, 1), pad=(0, 0),
                                   stride=stride,
                                   with_bias=False)
        s = F.average_pooling(x, (2, 2)) if dn else x
        return h + s

    def shift_unit(x, scope_name, dn=False):
        if module == "sc2":
            return sc2(x, scope_name, dn)

        if module == "csc":
            return csc(x, scope_name, dn)

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

    h = shift_unit(h, "conv2", False)    # -> 32x32
    h = shift_unit(h, "conv3", True)     # -> 16x16
    h = shift_unit(h, "conv4", False)    # -> 16x16
    h = shift_unit(h, "conv5", True)     # -> 8x8
    h = shift_unit(h, "conv6", False)    # -> 8x8
    h = shift_unit(h, "conv7", True)     # -> 4x4
    h = shift_unit(h, "conv8", False)    # -> 4x4
    h = F.average_pooling(h, kernel=(4, 4))  # -> 1x1
    pred = PF.affine(h, ncls)

    return pred
