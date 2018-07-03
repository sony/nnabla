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
from nnabla.monitor import Monitor, MonitorImageTile, MonitorSeries
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)
from nnabla.parametric_functions import parametric_function_api


@parametric_function_api("in")
def instance_normalization(inp, axes=[1], decay_rate=0.9, eps=1e-5,
                           batch_stat=True, output_stat=False, fix_parameters=False):
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
                                 decay_rate, eps, batch_stat, output_stat)


def convolution(x, n, kernel, stride, pad, init_method=None):
    if init_method == "paper":
        init = nn.initializer.NormalInitializer(0.02)
    else:
        s = nn.initializer.calc_normal_std_glorot(x.shape[1], n, kernel=kernel)
        init = nn.initializer.NormalInitializer(s)
    x = PF.convolution(x, n, kernel=kernel, stride=stride,
                       pad=pad, with_bias=True, w_init=init)
    return x


def deconvolution(x, n, kernel, stride, pad, init_method=None):
    if init_method == "paper":
        init = nn.initializer.NormalInitializer(0.02)
    else:
        s = nn.initializer.calc_normal_std_glorot(x.shape[1], n, kernel=kernel)
        init = nn.initializer.NormalInitializer(s)
    x = PF.deconvolution(x, n, kernel=kernel, stride=stride,
                         pad=pad, with_bias=True, w_init=init)
    return x


def convblock(x, n=0, k=(4, 4), s=(2, 2), p=(1, 1), leaky=False, init_method=None):
    x = convolution(x, n=n, kernel=k, stride=s, pad=p, init_method=init_method)
    x = instance_normalization(x, fix_parameters=True)
    x = F.leaky_relu(x, alpha=0.2) if leaky else F.relu(x)
    return x


def unpool_block(x, n=0, k=(4, 4), s=(2, 2), p=(1, 1), leaky=False, unpool=False, init_method=None):
    if not unpool:
        logger.info("Deconvolution was used.")
        x = deconvolution(x, n=n, kernel=k, stride=s,
                          pad=p, init_method=init_method)
    else:
        logger.info("Unpooling was used.")
        x = F.unpooling(x, kernel=(2, 2))
        x = convolution(x, n, kernel=(3, 3), stride=(1, 1),
                        pad=(1, 1), init_method=init_method)
    x = instance_normalization(x, fix_parameters=True)
    x = F.leaky_relu(x, alpha=0.2) if leaky else F.relu(x)
    return x


def resblock(x, n=256, init_method=None):
    r = x
    with nn.parameter_scope('block1'):
        r = convolution(r, n, kernel=(3, 3), pad=(1, 1),
                        stride=(1, 1), init_method=init_method)
        r = instance_normalization(r, fix_parameters=True)
        r = F.relu(r)
    with nn.parameter_scope('block2'):
        r = convolution(r, n, kernel=(3, 3), pad=(1, 1),
                        stride=(1, 1), init_method=init_method)
        r = instance_normalization(r, fix_parameters=True)
    return x + r


def generator(x, scopename, maps=64, unpool=False, init_method=None):
    with nn.parameter_scope('generator'):
        with nn.parameter_scope(scopename):
            with nn.parameter_scope('conv1'):
                x = convblock(x, n=maps, k=(7, 7), s=(1, 1), p=(3, 3),
                              leaky=False, init_method=init_method)
            with nn.parameter_scope('conv2'):
                x = convblock(x, n=maps*2, k=(3, 3), s=(2, 2), p=(1, 1),
                              leaky=False, init_method=init_method)
            with nn.parameter_scope('conv3'):
                x = convblock(x, n=maps*4, k=(3, 3), s=(2, 2), p=(1, 1),
                              leaky=False, init_method=init_method)
            for i in range(9):
                with nn.parameter_scope('res{}'.format(i+1)):
                    x = resblock(x, n=maps*4, init_method=init_method)
            with nn.parameter_scope('deconv1'):
                x = unpool_block(x, n=maps*2, k=(4, 4), s=(2, 2), p=(1, 1),
                                 leaky=False, unpool=unpool, init_method=init_method)
            with nn.parameter_scope('deconv2'):
                x = unpool_block(x, n=maps, k=(4, 4), s=(2, 2), p=(1, 1),
                                 leaky=False, unpool=unpool, init_method=init_method)
            with nn.parameter_scope('conv4'):
                x = convolution(x, 3, kernel=(7, 7), stride=(1, 1), pad=(3, 3),
                                init_method=init_method)
                x = F.tanh(x)
    return x


def discriminator(x, scopename, maps=64, init_method=None):
    with nn.parameter_scope('discriminator'):
        with nn.parameter_scope(scopename):
            with nn.parameter_scope('conv1'):
                x = convolution(x, maps, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
                                init_method=init_method)
                x = F.leaky_relu(x, alpha=0.2)
            with nn.parameter_scope('conv2'):
                x = convblock(x, n=maps*2, k=(4, 4), s=(2, 2), p=(1, 1),
                              leaky=True, init_method=init_method)
            with nn.parameter_scope('conv3'):
                x = convblock(x, n=maps*4, k=(4, 4), s=(2, 2), p=(1, 1),
                              leaky=True, init_method=init_method)
            with nn.parameter_scope('conv4'):
                x = convblock(x, n=maps*8, k=(4, 4), s=(1, 1), p=(1, 1),
                              leaky=True, init_method=init_method)
            with nn.parameter_scope('conv5'):
                x = convolution(x, 1, kernel=(4, 4), pad=(1, 1), stride=(1, 1),
                                init_method=init_method)
    return x


def f(x, unpool=False, init_method=None):
    return generator(x, scopename='y', unpool=unpool, init_method=init_method)


def g(y, unpool=False, init_method=None):
    return generator(y, scopename='x', unpool=unpool, init_method=init_method)


def d_x(x, init_method=None):
    return discriminator(x, scopename='x', init_method=init_method)


def d_y(y, init_method=None):
    return discriminator(y, scopename='y', init_method=init_method)


def image_augmentation(image):
    return F.image_augmentation(image,
                                shape=image.shape,
                                min_scale=1.0,
                                max_scale=286.0/256.0,  # == 1.1171875
                                flip_lr=True,
                                seed=rng_seed)


def recon_loss(x, y):
    return F.mean(F.absolute_error(x, y))


def lsgan_loss(d_fake, d_real=None, persistent=True):
    if d_real:  # Discriminator loss
        loss_d_real = F.mean(F.pow_scalar(d_real - 1., 2.))
        loss_d_fake = F.mean(F.pow_scalar(d_fake, 2.))
        loss = (loss_d_real + loss_d_fake) * 0.5
        loss.persistent = persistent
        return loss
    else:  # Generator loss, this form leads to minimization
        loss = F.mean(F.pow_scalar(d_fake - 1., 2.))
        loss.persistent = persistent
        return loss


def main():
    # Check generator's final output
    b, c, h, w = 4, 3, 256, 256
    x = nn.Variable([b, c, h, w])
    f_x = f(x)
    print(f_x.shape)

    y = nn.Variable([b, c, h, w])
    g_y = f(y)
    print(g_y.shape)

    # Check discriminator's final output
    d_x_var = d_x(f_x)
    print(d_x_var.shape)


if __name__ == '__main__':
    main()
