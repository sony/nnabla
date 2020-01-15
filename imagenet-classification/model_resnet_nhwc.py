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

"""
ResNet primitives and full network models.
"""

import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.initializer as I
from nnabla.logger import logger

import numpy as np

RNG = np.random.RandomState(214)


def pf_convolution(x, ochannels, kernel, stride=(1, 1), channel_last=False):
    axes = [3 if channel_last else 1]
    ichannels = x.shape[axes[0]]
    init = I.NormalInitializer(sigma=I.calc_normal_std_he_backward(
        ichannels, ochannels, kernel=kernel), rng=RNG)
    pad = tuple([int((k - 1) // 2) for k in kernel])
    return PF.convolution(x, ochannels, kernel, stride=stride, pad=pad,
                          with_bias=False, channel_last=channel_last,
                          w_init=init)


def pf_affine(r, num_classes=1000, channel_last=False):
    r = PF.convolution(r, num_classes, (1, 1), channel_last=channel_last,
                       w_init=I.NormalInitializer(sigma=0.01, rng=RNG), name='fc')
    return F.reshape(r, (r.shape[0], -1), inplace=False)


def shortcut(x, ochannels, stride, shortcut_type, test, channel_last=False):
    axes = [3 if channel_last else 1]
    ichannels = x.shape[axes[0]]
    use_conv = shortcut_type.lower() == 'c'
    if ichannels != ochannels:
        assert (ichannels * 2 == ochannels) or (ichannels * 4 == ochannels)
        if shortcut_type.lower() == 'b':
            use_conv = True
    if use_conv:
        # Convolution does everything.
        # Matching channels, striding.
        with nn.parameter_scope("shortcut_conv"):
            x = pf_convolution(x, ochannels, (1, 1),
                               stride=stride, channel_last=channel_last)
            x = PF.batch_normalization(x, axes=axes, batch_stat=not test)
    else:
        if stride != (1, 1):
            # Stride
            x = F.average_pooling(x, (1, 1), stride, channel_last=channel_last)
        if ichannels != ochannels:
            # Zero-padding to channel axis
            ishape = x.shape
            if channel_last:
                zero_shape = (ishape[0],) + ishape[1:3] + \
                              (ochannels - ichannels,)
            else:
                zero_shape = (ishape[0], ochannels - ichannels) + ishape[-2:]
            zeros = F.constant(zero_shape, 0)
            x = F.concatenate(x, zeros, axis=1)
    return x


def basicblock(x, ochannels, stride, shortcut_type, test, channel_last=False):
    def bn(h, z=None):
        axes = [3 if channel_last else 1]
        return PF.fused_batch_normalization(h, z, axes=axes, batch_stat=not test)

    conv_opts = dict(channel_last=channel_last)
    with nn.parameter_scope("basicblock1"):
        h = bn(pf_convolution(x, ochannels, (3, 3),
                              stride=stride, **conv_opts))

    with nn.parameter_scope("basicblock2"):
        h = pf_convolution(h, ochannels, (3, 3), **conv_opts)
    with nn.parameter_scope("basicblock_s"):
        s = shortcut(x, ochannels, stride, shortcut_type,
                     test, channel_last=channel_last)
    with nn.parameter_scope("basicblock2"):
        h = bn(h, s)
    return h


def bottleneck(x, ochannels, stride, shortcut_type, test, channel_last=False):
    def bn(h, z=None):
        axes = [3 if channel_last else 1]
        return PF.fused_batch_normalization(h, z, axes=axes, batch_stat=not test)

    conv_opts = dict(channel_last=channel_last)

    assert ochannels % 4 == 0
    hchannels = ochannels / 4
    with nn.parameter_scope("bottleneck1"):
        h = bn(pf_convolution(x, hchannels, (1, 1), **conv_opts))
    with nn.parameter_scope("bottleneck2"):
        h = bn(pf_convolution(h, hchannels, (3, 3),
                              stride=stride, **conv_opts))
    with nn.parameter_scope("bottleneck3"):
        h = pf_convolution(h, ochannels, (1, 1), **conv_opts)
    with nn.parameter_scope("bottleneck_s"):
        s = shortcut(x, ochannels, stride, shortcut_type,
                     test, channel_last=channel_last)
    with nn.parameter_scope("bottleneck3"):  # backward compat.
        h = bn(h, s)
    return h


def layer(x, block, ochannels, count, stride, shortcut_type, test, channel_last=False):
    for i in range(count):
        with nn.parameter_scope("layer{}".format(i + 1)):
            x = block(x, ochannels, stride if i ==
                      0 else (1, 1), shortcut_type, test, channel_last=channel_last)
    return x


def resnet_imagenet(x, num_classes, num_layers, shortcut_type, test, tiny=False, channel_last=False):
    """
    Args:
        x : Variable
        num_classes : Number of classes of outputs
        num_layers : Number of layers of ResNet chosen from (18, 34, 50, 101, 152)
        shortcut_type : 'c', 'b', ''
            'c' : Use Convolution anytime
            'b' : Use Convolution if numbers of channels of input
                  and output mismatch.
            '' : Use Identity mapping if channels match, otherwise zero padding.
        test : Construct net for testing.
        tiny (bool): Tiny imagenet mode. Input image must be (3, 56, 56).
        channel_last (bool):
            Channel dimmension comes at last in an input image. A.k.a NHWC order.
    """
    layers = {
        18: ((2, 2, 2, 2), basicblock, 1),
        34: ((3, 4, 6, 3), basicblock, 1),
        50: ((3, 4, 6, 3), bottleneck, 4),
        101: ((3, 4, 23, 3), bottleneck, 4),
        152: ((3, 8, 36, 3), bottleneck, 4)}

    counts, block, ocoef = layers[num_layers]
    logger.debug(x.shape)
    axes = [3 if channel_last else 1]
    with nn.parameter_scope("conv1"):
        stride = (1, 1) if tiny else (2, 2)
        r = pf_convolution(x, 64, (7, 7),
                           stride=stride,
                           channel_last=channel_last)
        r = PF.fused_batch_normalization(r, axes=axes, batch_stat=not test)
        r = F.max_pooling(r, (3, 3), stride, pad=(1, 1),
                          channel_last=channel_last)
    hidden = {}
    hidden['r0'] = r
    ochannels = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    logger.debug(r.shape)
    for i in range(4):
        with nn.parameter_scope("res{}".format(i + 1)):
            r = layer(r, block, ochannels[i] * ocoef,
                      counts[i], (strides[i], strides[i]), shortcut_type, test, channel_last=channel_last)
        hidden['r{}'.format(i + 1)] = r
        logger.debug(r.shape)
    pool_shape = r.shape[-2:]
    if channel_last:
        pool_shape = r.shape[1:3]
    r = F.average_pooling(r, pool_shape, channel_last=channel_last)
    with nn.parameter_scope("fc"):
        r = pf_affine(r, num_classes, channel_last=channel_last)
    return r, hidden
