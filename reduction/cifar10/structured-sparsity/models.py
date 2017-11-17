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

from collections import OrderedDict
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.parametric_functions import parametric_function_api
import os

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger
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


@parametric_function_api("conv")
def masked_convolution(inp, outmaps, kernel,
                       pad=None, stride=None, dilation=None, group=1,
                       w_init=None, b_init=None,
                       base_axis=1, fix_parameters=False, rng=None, with_bias=True):
    """
    """
    if w_init is None:
        w_init = UniformInitializer(
            calc_uniform_lim_glorot(inp.shape[base_axis], outmaps, tuple(kernel)), rng=rng)
    if with_bias and b_init is None:
        b_init = ConstantInitializer()
    w = get_parameter_or_create(
        "W", (outmaps, inp.shape[base_axis] / group) + tuple(kernel),
        w_init, not fix_parameters)
    mask_w = get_parameter_or_create("Mw", w.shape,
                                     ConstantInitializer(0.), False)
    w_masked = w * mask_w
    b = None
    b_masked = None
    if with_bias:
        b = get_parameter_or_create(
            "b", (outmaps,), b_init, not fix_parameters)
        mask_b = get_parameter_or_create("Mb", b.shape,
                                         ConstantInitializer(0.), False)
        b_masked = b * mask_b

    return F.convolution(inp, w_masked, b_masked,
                         base_axis, pad, stride, dilation, group)


def cifar10_resnet23_prediction(image, maps=64,
                                test=False):
    """
    Construct Resnet23.
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


def cifar10_resnet23_slim_prediction(image, maps=64,
                                     test=False):
    """
    Construct Resnet23.
    """
    # Residual Unit
    def res_unit(x, scope_name, rng, dn=False, test=False):
        C = x.shape[1]
        with nn.parameter_scope(scope_name):

            # Conv -> BN -> Relu
            with nn.parameter_scope("conv1"):
                h = masked_convolution(x, C / 2, kernel=(1, 1), pad=(0, 0),
                                       with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN -> Relu
            with nn.parameter_scope("conv2"):
                h = masked_convolution(h, C / 2, kernel=(3, 3), pad=(1, 1),
                                       with_bias=False)
                h = PF.batch_normalization(h, batch_stat=not test)
                h = F.relu(h)
            # Conv -> BN
            with nn.parameter_scope("conv3"):
                h = masked_convolution(h, C, kernel=(1, 1), pad=(0, 0),
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
        h = masked_convolution(image, maps, kernel=(3, 3), pad=(1, 1),
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


def filter_wise_reg(param):
    # L1 regularization over filter
    reg = F.sum(
        F.pow_scalar(F.sum(F.pow_scalar(param, 2), axis=[1, 2, 3]), 0.5))
    return reg


def channel_wise_reg(param):
    # L1 regularization over channel
    reg = F.sum(
        F.pow_scalar(F.sum(F.pow_scalar(param, 2), axis=[0, 2, 3]), 0.5))
    return reg


def ssl_regularization(params,
                       filter_decay,
                       channel_decay,
                       reg_type="both"):
    """Structured Sparsity Learning

    Wei Wen, et al.,
    "Learning Structured Sparsity in Deep Neural Networks",
    Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, 
    https://arxiv.org/abs/1608.03665.
    """
    reg = nn.Variable()
    reg.d = 0.
    for name, param in params.items():
        # SSL only on convolution weights
        if "conv/W" not in name:
            continue
        filt_reg = filter_wise_reg(param)
        chan_reg = channel_wise_reg(param)
        if reg_type == "filter":
            reg += filter_decay * filt_reg
        elif reg_type == "channel":
            reg += channel_decay * chan_reg
        elif reg_type == "both":
            reg += filter_decay * filt_reg + channel_decay * chan_reg
    return reg


def create_and_set_mask(params, rrate=0.5):
    # Compute L2 Norm
    l2_norms = OrderedDict()
    for name, param in params.items():
        # SSL only on convolution weights
        if not ("conv/W" in name or "conv/b" in name):
            continue
        l2_norm = np.sqrt(np.sum(param.d ** 2, axis=(1, 2, 3)))
        l2_norms[name] = l2_norm  # for each filter

    # Sort index by l2-norm, then create mask
    masks = OrderedDict()
    for name, l2_norm in l2_norms.items():
        idx_sorted = sorted(range(len(l2_norm)),
                            key=lambda k: l2_norm[k],
                            )
        idx_sorted_reduced = idx_sorted[:int(len(idx_sorted) * rrate)]
        x = np.ones_like(idx_sorted)
        x[idx_sorted_reduced] = 0.
        mask_data = np.broadcast_to(
            x.reshape((len(idx_sorted), 1, 1, 1)),
            params[name].shape)
        if "/W" in name:
            masks[name.replace("/W", "/Mw")] = mask_data
        if "/b" in name:
            masks[name.replace("/b", "/Mb")] = mask_data

    # Set mask
    for name, param in params.items():
        if not ("/Mw" in name or "/Mb" in name):
            continue
        param.d = masks[name]
