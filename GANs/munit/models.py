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
import numpy as np
import argparse
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorImageTile, MonitorSeries
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot, calc_normal_std_he_backward,
    ConstantInitializer, NormalInitializer, UniformInitializer)
from nnabla.parametric_functions import parametric_function_api


@parametric_function_api("bn")
def BN(inp, axes=[1], decay_rate=0.9, eps=1e-5,
       batch_stat=True, output_stat=False, fix_parameters=False):
    """Batch Normalization
    """
    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat, output_stat)


@parametric_function_api("in")
def IN(inp, axes=[1], decay_rate=0.9, eps=1e-5, fix_parameters=True):
    """Instance Normalization
    """
    if inp.shape[0] == 1:
        return INByBatchNorm(inp, axes, decay_rate, eps, fix_parameters)

    b, c = inp.shape[0:2]
    spacial_shape = inp.shape[2:]

    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)

    # Instance normalization
    # normalize over spatial dimensions
    axis = [i for i in range(len(inp.shape)) if i > 1]
    mean = F.sum(inp, axis=axis, keepdims=True) / np.prod(axis)
    var = F.pow_scalar(F.sum(inp - mean, axis=axis,
                             keepdims=True), 2.0) / np.prod(axis)
    h = (inp - mean) / F.pow_scalar(var + eps, 0.5)
    return gamma * inp + beta


@parametric_function_api("in")
def INByBatchNorm(inp, axes=[1], decay_rate=0.9, eps=1e-5, fix_parameters=True):
    """Instance Normalization (implemented using BatchNormalization)
    Instance normalization is equivalent to the batch normalization if a batch size is one, in
    other words, it normalizes over spatial dimension(s), meaning all dimensions except for
    the batch and feature dimension.
    """
    assert len(axes) == 1

    shape_stat = [1 for _ in inp.shape]
    shape_stat[axes[0]] = inp.shape[axes[0]]
    beta = get_parameter_or_create(
        "beta", shape_stat, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", shape_stat, ConstantInitializer(1), not fix_parameters)
    mean = get_parameter_or_create(
        "mean", shape_stat, ConstantInitializer(0), False)
    var = get_parameter_or_create(
        "var", shape_stat, ConstantInitializer(0), False)
    return F.batch_normalization(inp, beta, gamma, mean, var, axes,
                                 decay_rate, eps, batch_stat=True, output_stat=False)


@parametric_function_api("adain")
def AdaIN(h, style, fix_parameters=False):
    """Adaptive Instance Normalization
    """
    b, c, _, _ = h.shape
    gamma = style[:, :c].reshape([b, c, 1, 1])
    beta = style[:, c:].reshape([b, c, 1, 1])
    h = IN(h, name="in", fix_parameters=True)
    h = gamma * h + beta
    return h


def f_layer_normalization(inp, beta, gamma, eps=1e-5):
    use_axis = [x for x in range(1, inp.ndim)]
    inp = F.sub2(inp, F.mean(inp, axis=use_axis, keepdims=True))
    inp = F.div2(inp, F.pow_scalar(
        F.mean(F.pow_scalar(inp, 2), axis=use_axis, keepdims=True), 0.5) + eps)
    return inp * F.broadcast(gamma, inp.shape) + F.broadcast(beta, inp.shape)


@parametric_function_api("ln")
def LN(inp, fix_parameters=False):
    """Layer normalization.
    """
    beta_shape = (1, inp.shape[1], 1, 1)
    gamma_shape = (1, inp.shape[1], 1, 1)
    beta = get_parameter_or_create(
        "beta", beta_shape, ConstantInitializer(0), not fix_parameters)
    gamma = get_parameter_or_create(
        "gamma", gamma_shape, ConstantInitializer(1), not fix_parameters)
    return f_layer_normalization(inp, beta, gamma)


def convolution(x, maps, kernel=(3, 3), pad=(0, 0, 0, 0), stride=(1, 1),
                pad_mode="reflect", name="conv"):
    """Convolution wapper"""
    if type(kernel) == int:
        kernel = tuple([kernel] * 2)
    if type(pad) == int:
        pad = tuple([pad] * 4)
    if type(stride) == int:
        stride = tuple([stride] * 2)

    h = x
    #s = nn.initializer.calc_normal_std_glorot(h.shape[1], maps, kernel=kernel)
    s = nn.initializer.calc_normal_std_he_backward(
        h.shape[1], maps, kernel=kernel)
    init = nn.initializer.NormalInitializer(s)
    h = F.pad(h, pad, mode=pad_mode)
    h = PF.convolution(h, maps, kernel, stride=stride,
                       with_bias=True, w_init=init, name=name)
    return h


def normalize(x, style=None, norm=""):
    h = x
    if norm == "in":
        h = IN(h, fix_parameters=True)
    elif norm == "adain":
        h = AdaIN(h, style, fix_parameters=True)
    elif norm == "ln":
        h = LN(h, fix_parameters=False)
    else:
        h = h
    return h


def convblock(x, maps, kernel=(3, 3), pad=(0, 0, 0, 0), stride=1,
              with_bias=True,
              pad_mode="reflect", norm="", leaky=False,
              name="convblock"):
    """Convolution block"""
    h = x
    with nn.parameter_scope(name):
        # conv -> norm -> act
        h = convolution(h, maps, kernel, pad, stride, pad_mode)
        h = normalize(h, norm=norm)
        h = F.relu(h, True) if not leaky else F.leaky_relu(h, 0.2, True)
    return h


def mlp(x, maps, num_res=4, num_layers=2, name="mlp"):
    h = x
    with nn.parameter_scope(name):
        h = PF.affine(h, maps, name="affine-first")
        h = F.relu(h, True)
        h = PF.affine(h, maps, name="affine-mid")
        h = F.relu(h, True)
        h = PF.affine(h, 2 * maps * num_res * num_layers, name="affine-last")
    return h


def style_encoder(x, maps=64, name="style-encoder"):
    h = x
    with nn.parameter_scope("generator"):
        with nn.parameter_scope(name):
            h = convblock(h, maps * 1, 7, 3, 1, norm="", name="convblock-1")
            h = convblock(h, maps * 2, 4, 1, 2, norm="", name="convblock-2")
            h = convblock(h, maps * 4, 4, 1, 2, norm="", name="convblock-3")
            h = convblock(h, maps * 4, 4, 1, 2, norm="", name="convblock-4")
            h = convblock(h, maps * 4, 4, 1, 2, norm="", name="convblock-5")
            h = F.average_pooling(h, h.shape[2:])
            h = convolution(h, maps * 4, 1, 0, 1)
    return h


def content_encoder(x, maps=64, pad_mode="reflect", name="content-encoder"):
    h = x
    with nn.parameter_scope("generator"):
        with nn.parameter_scope(name):
            h = convblock(h, maps * 1, 7, 3, 1, norm="in",
                          pad_mode=pad_mode, name="convblock-1")
            h = convblock(h, maps * 2, 4, 1, 2, norm="in",
                          pad_mode=pad_mode, name="convblock-2")
            h = convblock(h, maps * 4, 4, 1, 2, norm="in",
                          pad_mode=pad_mode, name="convblock-3")
            h = resblock(h, None, maps * 4, norm="in",
                         pad_mode=pad_mode, name="resblock-1")
            h = resblock(h, None, maps * 4, norm="in",
                         pad_mode=pad_mode, name="resblock-2")
            h = resblock(h, None, maps * 4, norm="in",
                         pad_mode=pad_mode, name="resblock-3")
            h = resblock(h, None, maps * 4, norm="in",
                         pad_mode=pad_mode, name="resblock-4")
    return h


def resblock(x, style=None, maps=256, pad_mode="reflect", norm="", name="resblock"):
    h = x

    def style_func(pos):
        if style is None:
            return None
        return style[:, pos*maps*2:(pos+1)*maps*2]
    with nn.parameter_scope(name):
        with nn.parameter_scope("conv-1"):
            h = convolution(h, maps, 3, 1, 1, pad_mode=pad_mode)
            h = normalize(h, style_func(0), norm)
            h = F.relu(h, True)
        with nn.parameter_scope("conv-2"):
            h = convolution(h, maps, 3, 1, 1, pad_mode=pad_mode)
            h = normalize(h, style_func(1), norm)
    return h + x


def decoder(content, style, maps=256, num_res=4, num_layers=2, pad_mode="reflect", name="decoder"):
    h = content
    styles = mlp(style, maps, num_res, num_layers)
    b, c, _, _ = h.shape
    with nn.parameter_scope("generator"):
        with nn.parameter_scope(name):
            for i in range(num_res):
                s = styles[:, i*maps*num_layers*2:(i+1)*maps*num_layers*2]
                h = resblock(h, s, maps, norm="adain", pad_mode=pad_mode,
                             name="resblock-{}".format(i + 1))
            h = upsample(h, maps // 2, norm="ln",
                         pad_mode=pad_mode, name="upsample-1")
            h = upsample(h, maps // 4, norm="ln",
                         pad_mode=pad_mode, name="upsample-2")
            h = convolution(h, 3, 7, 3, 1, pad_mode=pad_mode, name="to-RGB")
            h = F.tanh(h)
    return h


def upsample(x, maps, norm="ln", pad_mode="reflect", name="upsample"):
    h = x
    with nn.parameter_scope(name):
        #h = F.interpolate(h, (2, 2), mode="linear")
        h = F.unpooling(h, (2, 2))
        h = convblock(h, maps, 5, 2, 1, norm=norm, pad_mode=pad_mode)
    return h


def discriminator(x, maps=64, name="discriminator"):
    h = x
    with nn.parameter_scope(name):
        h = convblock(h, maps * 1, 4, 1, 2, leaky=True, name="convblock-1")
        h = convblock(h, maps * 2, 4, 1, 2, leaky=True, name="convblock-2")
        h = convblock(h, maps * 4, 4, 1, 2, leaky=True, name="convblock-3")
        h = convblock(h, maps * 8, 4, 1, 2, leaky=True, name="convblock-4")
        h = convolution(h, 1, 1, 0, 1, name="last-conv")
    return h


def discriminators(x, maps=64, n=3):
    h = x
    discriminators = []
    with nn.parameter_scope("discriminators"):
        for i in range(n):
            h = discriminator(x, maps, name="discriminator-{}x".format(2 ** i))
            discriminators.append(h)
            x = F.average_pooling(x, kernel=(3, 3), stride=(
                2, 2), pad=(1, 1), including_pad=False)
    return discriminators


def recon_loss(x, y):
    return F.mean(F.absolute_error(x, y))


def lsgan_loss(d_fake, d_real=None, persistent=True):
    if d_real:  # Discriminator loss
        loss_d_real = F.mean((d_real - 1.0) ** 2.0)
        loss_d_fake = F.mean(d_fake ** 2.0)
        loss = loss_d_real + loss_d_fake
        return loss
    else:  # Generator loss, this form leads to minimization
        loss = F.mean((d_fake - 1.0) ** 2.0)
        return loss


def main():
    # Input
    b, c, h, w = 1, 3, 256, 256
    x_real = nn.Variable([b, c, h, w])

    # Conent Encoder
    content = content_encoder(
        x_real, maps=64, pad_mode="reflect", name="content-encoder")
    print("Content shape: ", content.shape)

    # Style Encoder
    style = style_encoder(x_real, maps=64, name="style-encoder")
    print("Style shape: ", style.shape)

    # Decoder
    x_fake = decoder(content, style, name="decoder")
    print("X_fake shape: ", x_fake.shape)

    for k, v in nn.get_parameters().items():
        if "gamma" in k or "beta" in k:
            print(k, np.prod(v.shape))

    # Discriminator
    p_reals = discriminators(x_real)
    for i, p_real in enumerate(p_reals):
        print("Scale: ", i, "p_fake shape: ", p_real.shape)


if __name__ == '__main__':
    main()
