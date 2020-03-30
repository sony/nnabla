# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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


def get_channel_axis(channel_last):
    return 3 if channel_last else 1


def get_spatial_axes(channel_last):
    return (1, 2) if channel_last else (2, 3)


def get_spatial_shape(shape, channel_last):
    return [shape[a] for a in get_spatial_axes(channel_last)]


def pf_convolution(x, ochannels, kernel, stride=(1, 1), group=1, channel_last=False, with_bias=False):
    axes = [get_channel_axis(channel_last)]
    ichannels = x.shape[axes[0]]
    init = I.NormalInitializer(sigma=I.calc_normal_std_he_forward(
        ichannels, ochannels, kernel=kernel), rng=RNG)
    pad = tuple([int((k - 1) // 2) for k in kernel])
    return PF.convolution(x, ochannels, kernel, stride=stride, pad=pad, group=group,
                          with_bias=with_bias, channel_last=channel_last,
                          w_init=init)


def pf_affine(r, num_classes=1000, channel_last=False):
    # Initializer supposes the final classifaction layer
    fan_in = int(np.prod(r.shape[1:]))
    k = 1 / np.sqrt(fan_in)
    init = I.UniformInitializer((-k, k), rng=RNG)
    r = PF.convolution(r, num_classes, (1, 1), channel_last=channel_last,
                       w_init=init, b_init=init, name='fc')
    return F.reshape(r, (r.shape[0], -1), inplace=False)


def shortcut(x, ochannels, stride, shortcut_type, test, channel_last=False):
    '''
    Defines operations used in residual connection in ResNet.

    Args:
        x (Variable):
            Variable with NHWC or NCHW memory layout which must be
            consistent with`channel_last`.
        stride (tuple of 2 ints):
            Spcifies stride factor for each spatial dimension.
        shortcut_type (str):
            Specifies shortcut type from 'c', 'b', and ''. The default is 'b'.
            * '': Zero-padding is applied if input channels doesn't match
              `ochannels`.
            * 'b': Convolution is applied which outputs `ochannels`
              if input channels doesn't match `ochannels`.
            * 'c': Convolution is always applied which outputs `ochannels`.
        test (bool):
            Construct net for testing. For example with True, batch norm is
            created as batch stat mode (batch_stat=True).
        channel_last (bool):
            If channel_last is True, Channel dimmension must come at last in
            an input image. A.k.a NHWC order.

    Returns: Variable

    '''
    axes = [get_channel_axis(channel_last)]
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


class BasicBlock(object):

    '''
    Basic block used at cells in ResNet-18, and 34.

    See `Bottleneck` for argument details of `__init__`.

    '''

    def __init__(self, shortcut_type='b', test=True, channel_last=False):
        self.shortcut_type = shortcut_type
        self.test = test
        self.channel_last = channel_last
        self.conv_opts = dict(channel_last=self.channel_last)

    def bn(self, h, z=None, no_relu=False):
        axes = [get_channel_axis(self.channel_last)]
        if no_relu:
            h = PF.batch_normalization(h, axes=axes, batch_stat=not self.test)
            if z is None:
                return h
            return F.add2(z, h, inplace=True)
        return PF.fused_batch_normalization(h, z, axes=axes, batch_stat=not self.test)

    def __call__(self, x, ochannels, stride):
        '''
        Defines a basic block.

        See 'Bottleneck.__call__()` for argument details.

        '''

        with nn.parameter_scope("basicblock1"):
            h = self.bn(pf_convolution(x, ochannels, (3, 3),
                                       stride=stride, **self.conv_opts))

        with nn.parameter_scope("basicblock2"):
            h = pf_convolution(h, ochannels, (3, 3), **self.conv_opts)
        with nn.parameter_scope("basicblock_s"):
            s = shortcut(x, ochannels, stride, self.shortcut_type,
                         self.test, channel_last=self.channel_last)
        with nn.parameter_scope("basicblock2"):
            h = self.bn(h, s)
        return h


class BottleneckBlock(BasicBlock):
    '''
    Bottleneck block which is used at cells in ResNet-50, 101, and 150.

    Args:
        shortcut_type (str):
            Specifies shortcut type from 'c', 'b', and ''. The default is 'b'.
            See `shortcut()` for details.
        test (bool): See `ResNetBase`.
        channel_last: See `ResNetBase`.


    '''

    def __init__(self, shortcut_type='b', test=True, channel_last=False):
        super(BottleneckBlock, self).__init__(
            shortcut_type, test, channel_last)

    def __call__(self, x, ochannels, stride):
        '''
        Define a bottleneck block.

        Args:
            x (Variable):
                Variable with NHWC or NCHW memory layout which must be
                consistent with`self.channel_last`.
            ochannels (int): Number of output channels of this block.
            stride (tuple of 2 ints):
                Spcifies stride factor for each spatial dimension.

        Returns: Variable

        '''
        assert ochannels % 4 == 0
        hchannels = ochannels // 4
        with nn.parameter_scope("bottleneck1"):
            h = self.bn(pf_convolution(x, hchannels, (1, 1), **self.conv_opts))
        with nn.parameter_scope("bottleneck2"):
            h = self.bn(pf_convolution(h, hchannels, (3, 3),
                                       stride=stride, **self.conv_opts))
        with nn.parameter_scope("bottleneck3"):
            h = pf_convolution(h, ochannels, (1, 1), **self.conv_opts)
        with nn.parameter_scope("bottleneck_s"):
            s = shortcut(x, ochannels, stride, self.shortcut_type,
                         self.test, channel_last=self.channel_last)
        with nn.parameter_scope("bottleneck3"):  # backward compat.
            h = self.bn(h, s)
        return h


class ResNetBase(object):

    '''
    A class for defining ResNet-like architecture.

    Architecture overview:

    * 7x7 conv + BN + max pooling
      * downsampling is applied to halve the spatial dimensions twice
        each at conv and pooling. Hence, after this part, the spatial
        size is reduced by 4.
    * A sequence of "cell"s (4 cells in this class)
      * A "cell" consists of a sqeuence of "block"s which is specified as an
        argument `block`. By default, block types for standard ResNet are used
        according to number of layers. See `get_default_block_type()`. For
        example in ResNet50 (i.e. `num_layers=50`), `BottleneckBlock`s are
        applied repeatly 3 times for the first cell, then 4, 6, and 3 for 2nd,
        3rd, and 4th cell respectively. The numbers of blocks applied for cells
        are determined by returned values of a member method `get_num_blocks()`.
      * At the first block of each cell, strided convolution is used to reduce
        the spatial size by a factor specified by the returned value of a
        member method `get_cell_configuration()`.
      * Number of output channels for each block is also determined by the
        returned values of `get_cell_configuration()`.
    * Global average pooling + FC (fully connected layer) for classification.

    Args:
        num_classes (int): Number of classes of outputs
        num_layers (int):
            Number of layers of ResNet chosen from (18, 34, 50, 101, 152)
        max_pooling_ceil_border (bool):
            At the first max pooling after the first conv+bn,
            `ignore_border=False` is used which is equivalent to Caffe's CEIL
            rounding mode, if True. Otherwise, `pad=(1, 1)` is used.
        block (block type):
            A class object which implements a callable with arguments as follows.
                * x: a `Variable` object.
                * ochannels: output channels of the block.
                * stride: a tuple of 2 ints which specifies the spatial size
                  reduction factor applied to an input tensor of a block.

            See `BasicBlock` and `BottleneckBlock` for implementations.

        test (bool):
            Construct net for testing. For example with True, batch norm is
            created as batch stat mode (batch_stat=True).
        channel_last (bool):
            If channel_last is True, Channel dimmension comes at last in an input image. A.k.a NHWC order.
    '''

    _default_num_blocks = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }

    def __init__(self, num_classes=1000, num_layers=50, max_pooling_ceil_border=False, block=None, test=True, channel_last=False):
        assert num_layers in self._default_num_blocks
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.max_pooling_ceil_border = max_pooling_ceil_border
        self.test = test
        self.channel_last = channel_last
        if block is None:
            Block = self.get_default_block_type(num_layers)
            block = Block('b', self.test, self.channel_last)
        self.block = block

    def get_num_blocks(self, num_layers):
        return self._default_num_blocks[num_layers]

    def get_default_block_type(self, num_layers):
        if num_layers < 50:
            return BasicBlock
        return BottleneckBlock

    def get_cell_configurations(self, num_layers):
        ochannels = [64, 128, 256, 512]
        if num_layers >= 50:
            ochannels = list(map(lambda x: x * 4, ochannels))
        strides = [1, 2, 2, 2]
        return (
            self.get_num_blocks(num_layers),
            ochannels,
            strides)

    def cell(self, x, ochannels, count, stride):
        '''
        Create a cell where blocks are repeatedly applied `count` times.
        '''
        for i in range(count):
            with nn.parameter_scope("layer{}".format(i + 1)):
                x = self.block(x, ochannels, stride if i == 0 else (1, 1))
        return x

    def __call__(self, x):
        '''
        Defines a ResNet-like network according to the configuration specified.

        Args:
            x:
                A Variable object which has a shape with a format
                `NCHW` if `channel_last=False` else `NHWC`.

        Returns:
            * An output `Variable` of classification layer
            * Intermediate `Variable` outputs from input and output of each
              cell

        '''

        logger.debug(x.shape)

        # First convolution
        axes = [get_channel_axis(self.channel_last)]
        with nn.parameter_scope("conv1"):
            r = pf_convolution(x, 64, (7, 7), stride=(2, 2),
                               channel_last=self.channel_last)
            r = PF.fused_batch_normalization(
                r, axes=axes, batch_stat=not self.test)
            mp_opts = dict(
                ignore_border=False) if self.max_pooling_ceil_border else dict(pad=(1, 1))
            r = F.max_pooling(r, (3, 3), (2, 2),
                              channel_last=self.channel_last, **mp_opts)
        hidden = {}
        hidden['r0'] = r
        logger.debug(r.shape)

        # Create cells each of which consists of blocks repeatedly applied
        cell_configs = self.get_cell_configurations(self.num_layers)
        for i, (counts, ochannels, strides) in enumerate(zip(*cell_configs)):
            with nn.parameter_scope("res{}".format(i + 1)):
                r = self.cell(r, ochannels, counts, (strides,) * 2)
            hidden['r{}'.format(i + 1)] = r
            logger.debug(r.shape)

        # Global average pooling
        pool_shape = get_spatial_shape(r.shape, self.channel_last)
        r = F.average_pooling(r, pool_shape, channel_last=self.channel_last)

        # Final classification layer
        with nn.parameter_scope("fc"):
            r = pf_affine(r, self.num_classes, channel_last=self.channel_last)
        return r, hidden
