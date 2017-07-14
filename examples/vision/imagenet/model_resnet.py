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
from nnabla.logger import logger


def shortcut(x, ochannels, stride, shortcut_type, test):
    ichannels = x.shape[1]
    use_conv = shortcut_type.lower() == 'c'
    if ichannels != ochannels:
        assert (ichannels * 2 == ochannels) or (ichannels * 4 == ochannels)
        if shortcut_type.lower() == 'b':
            use_conv = True
    if use_conv:
        # Convolution does everything.
        # Matching channels, striding.
        with nn.parameter_scope("shortcut_conv"):
            x = PF.convolution(x, ochannels, (1, 1),
                               stride=stride, with_bias=False)
            x = PF.batch_normalization(x, batch_stat=not test)
    else:
        if stride != (1, 1):
            # Stride
            x = F.average_pooling(x, (1, 1), stride)
        if ichannels != ochannels:
            # Zero-padding to channel axis
            ishape = x.shape
            zeros = F.constant(
                0, (ishape[0], ochannels - ichannels) + ishape[-2:])
            x = F.concatenate(x, zeros, axis=1)
    return x


def basicblock(x, ochannels, stride, shortcut_type, test):
    def bn(h):
        return PF.batch_normalization(h, batch_stat=not test)
    ichannels = x.shape[1]
    with nn.parameter_scope("basicblock1"):
        h = F.relu(bn(PF.convolution(x, ochannels, (3, 3),
                                     pad=(1, 1), stride=stride, with_bias=False)),
                   inplace=True)
    with nn.parameter_scope("basicblock2"):
        h = bn(PF.convolution(h, ochannels, (3, 3), pad=(1, 1), with_bias=False))
    with nn.parameter_scope("basicblock_s"):
        s = shortcut(x, ochannels, stride, shortcut_type, test)
    return F.relu(F.add2(h, s, inplace=True), inplace=True)


def bottleneck(x, ochannels, stride, shortcut_type, test):
    def bn(h):
        return PF.batch_normalization(h, batch_stat=not test)
    assert ochannels % 4 == 0
    hchannels = ochannels / 4
    with nn.parameter_scope("bottleneck1"):
        h = F.relu(
            bn(PF.convolution(x, hchannels, (1, 1), with_bias=False)),
            inplace=True)
    with nn.parameter_scope("bottleneck2"):
        h = F.relu(
            bn(PF.convolution(h, hchannels, (3, 3), pad=(1, 1),
                              stride=stride, with_bias=False)), inplace=True)
    with nn.parameter_scope("bottleneck3"):
        h = bn(PF.convolution(h, ochannels, (1, 1), with_bias=False))
    with nn.parameter_scope("bottleneck_s"):
        s = shortcut(x, ochannels, stride, shortcut_type, test)
    return F.relu(F.add2(h, s, inplace=True), inplace=True)


def layer(x, block, ochannels, count, stride, shortcut_type, test):
    for i in range(count):
        with nn.parameter_scope("layer{}".format(i + 1)):
            x = block(x, ochannels, stride if i ==
                      0 else (1, 1), shortcut_type, test)
    return x


def resnet_imagenet(x, num_classes, num_layers, shortcut_type, test, tiny=False):
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
    """
    layers = {
        18: ((2, 2, 2, 2), basicblock, 1),
        34: ((3, 4, 6, 3), basicblock, 1),
        50: ((3, 4, 6, 3), bottleneck, 4),
        101: ((3, 4, 23, 3), bottleneck, 4),
        152: ((3, 8, 36, 3), bottleneck, 4)}

    counts, block, ocoef = layers[num_layers]
    logger.debug(x.shape)
    with nn.parameter_scope("conv1"):
        stride = (1, 1) if tiny else (2, 2)
        r = PF.convolution(x, 64, (7, 7),
                           pad=(3, 3), stride=stride, with_bias=False)
        r = F.relu(PF.batch_normalization(
            r, batch_stat=not test), inplace=True)
        r = F.max_pooling(r, (3, 3), stride, False)
    hidden = {}
    hidden['r0'] = r
    ochannels = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    logger.debug(r.shape)
    for i in range(4):
        with nn.parameter_scope("res{}".format(i + 1)):
            r = layer(r, block, ochannels[i] * ocoef,
                      counts[i], (strides[i], strides[i]), shortcut_type, test)
        hidden['r{}'.format(i + 1)] = r
        logger.debug(r.shape)
    r = F.average_pooling(r, r.shape[-2:])
    with nn.parameter_scope("fc"):
        r = PF.affine(r, num_classes)
    logger.debug(r.shape)
    return r, hidden
