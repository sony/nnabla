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
import numpy as np
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
from nnabla.parameter import get_parameter_or_create
from nnabla.ext_utils import get_extension_context
from nnabla.parametric_functions import parametric_function_api

from nnabla.initializer import (
    calc_uniform_lim_glorot, calc_normal_std_he_backward, calc_normal_std_he_forward,
    ConstantInitializer, NormalInitializer, UniformInitializer)

from functools import reduce


def upsample(h, maps, up, test=False, name="convblock"):
    if up == "nearest":
        h = PF.convolution(h, maps, (3, 3), (1, 1), name=name)
        h = F.interpolate(h, scale=(2, 2), mode="nearest")
    elif up == "linear":
        h = PF.convolution(h, maps, (3, 3), (1, 1), name=name)
        h = F.interpolate(h, scale=(2, 2), mode="linear")
    elif up == "unpooling":
        h = PF.convolution(h, maps, (3, 3), (1, 1), name=name)
        h = F.unpooling(h, (2, 2))
    elif up == "deconv":
        h = PF.deconvolution(h, maps * 2, (2, 2), (0, 0), (2, 2), name=name)
    else:
        raise ValueError(
            'Set "up" option in ["nearest", "linear", "unpooling", "deconv"]')
    h = PF.batch_normalization(h, batch_stat=not test, name=name)
    h = F.relu(h)

    return h


def generator(z, scopename="generator", maps=128, s=4, test=False, up="nearest"):
    b, _ = z.shape
    with nn.parameter_scope(scopename):
        # Project
        h = PF.affine(z, maps * 4 * s * s)
        h = F.reshape(h, [b, maps * 4, s, s])
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
        # Convblocks
        h = upsample(h, maps * 4, up, test, name="convblock-1")  # 4x4 -> 8x8
        h = upsample(h, maps * 2, up, test, name="convblock-2")  # 8x8 -> 16x16
        h = upsample(h, maps * 1, up, test,
                     name="convblock-3")  # 16x16 -> 32x32
        # Last conv
        h = PF.convolution(h, 3, (3, 3), (1, 1), name="to-RGB")
        h = F.tanh(h)
    return h


def discriminator(x, scopename="discriminator", maps=128):
    with nn.parameter_scope(scopename):
        h = PF.convolution(x, maps * 1, (3, 3), name="conv-1")
        h = F.average_pooling(h, (2, 2))      # 32x32 -> 16x16
        h = F.leaky_relu(h, 0.01)

        h = PF.convolution(h, maps * 2, (3, 3), name="conv-2")
        h = F.average_pooling(h, (2, 2))      # 16x16 -> 8x8
        h = F.leaky_relu(h, 0.01)

        h = PF.convolution(h, maps * 4, (3, 3), name="conv-3")
        h = F.average_pooling(h, (2, 2))      # 8x8 -> 4x4
        h = F.leaky_relu(h, 0.01)
        h = PF.affine(h, 1)

    return h


def gan_loss(p_fake, p_real=None):
    if p_real is not None:
        return F.mean(p_fake) - F.mean(p_real)
    return -F.mean(p_fake)


if __name__ == '__main__':
    # Config
    b, c, h, w = 4, 3, 32, 32
    latent = 128
    eps = np.random.rand()
    ctx = get_extension_context("cudnn")
    nn.set_default_context(ctx)

    z = nn.Variable.from_numpy_array(np.random.randn(b, latent))
    x_real = nn.Variable.from_numpy_array(
        np.random.randn(b, c, h, w)) / 127.5 - 1.0

    # Fake sample
    print("# Fake sample")
    x_fake = generator(z, test=False)
    print(x_fake)

    # Prob for fake sample
    print("# Prob for fake sample")
    p_fake = discriminator(x_fake)
    print(p_fake)

    # Prob for real sample
    p_real = discriminator(x_real)

    # WGAN loss
    print("# WGAN loss")
    loss_gen = gan_loss(p_fake)
    print(loss_gen)
    loss_dis = gan_loss(p_fake, p_real)
    print(loss_dis)

    # Gradient penalty
    print("# Gradient penalty")
    x_rmix = eps * x_real + (1.0 - eps) * x_fake
    p_rmix = discriminator(x_rmix)
    grads = nn.grad([p_rmix], [x_rmix])
    print(grads)
    l2norms = [F.sum(g ** 2.0, [1, 2, 3]) ** 0.5 for g in grads]
    gp = sum([F.mean((l - 1.0) ** 2.0) for l in l2norms])

    loss_dis += gp
    gp.forward()
    gp.backward()
