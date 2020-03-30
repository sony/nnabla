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

import nnabla as nn
import nnabla.functions as F

from .base import (
    ResNetBase,
    BottleneckBlock,
    pf_convolution,
    get_channel_axis,
    get_spatial_shape,
    shortcut,
)

from . import registry


class SEBlock(object):
    '''
    Squeeze-and-Excitation block.

    Args:
        reduction_ratio (int):
            A number of channels is reduced by this ratio at the first
            convolution in SE block.
        channel_last (bool): See `ResNetBase`.

    '''

    def __init__(self, reduction_ratio=16, channel_last=False):
        self.reduction_ratio = reduction_ratio
        self.channel_last = channel_last

    def __call__(self, x):
        conv_opts = dict(channel_last=self.channel_last, with_bias=True)
        channel_axis = get_channel_axis(self.channel_last)
        nc = x.shape[channel_axis]
        nc_r = nc // self.reduction_ratio

        # Squeeze
        pool_shape = get_spatial_shape(x.shape, self.channel_last)
        s = F.average_pooling(x, pool_shape, channel_last=self.channel_last)
        with nn.parameter_scope('fc1'):
            s = F.relu(pf_convolution(s, nc_r, (1, 1), **conv_opts))
        with nn.parameter_scope('fc2'):
            s = F.sigmoid(pf_convolution(s, nc, (1, 1), **conv_opts))

        # Excitation
        return x * s


class SEBottleneckBlock(BottleneckBlock):
    '''
    Bottleneck block with Squeeze-and-Excitation block.

    Args:
        resnext (bool): See `SENet`.
        reduction_ratio (int):  See `SEBlock`.

    See `BottleneckBlock` for other details.

    '''

    def __init__(self, resnext=False, reduction_ratio=16, shortcut_type='b', test=True, channel_last=False):
        super(SEBottleneckBlock, self).__init__(
            shortcut_type, test, channel_last)
        self.resnext = resnext
        self.seblock = SEBlock(reduction_ratio, channel_last=channel_last)

    def get_bottleneck_channels(self, ochannels):
        div = 2 if self.resnext else 4
        assert ochannels % div == 0
        return ochannels // div

    def __call__(self, x, ochannels, stride):

        hchannels = self.get_bottleneck_channels(ochannels)
        cardinality = 32 if self.resnext else 1
        stride1 = None if self.resnext else stride
        stride2 = stride if self.resnext else None

        with nn.parameter_scope("bottleneck1"):
            h = self.bn(pf_convolution(x, hchannels, (1, 1),
                                       stride=stride1, **self.conv_opts))
        with nn.parameter_scope("bottleneck2"):
            h = self.bn(pf_convolution(h, hchannels, (3, 3), stride=stride2,
                                       group=cardinality, **self.conv_opts))
        with nn.parameter_scope("bottleneck3"):
            h = pf_convolution(h, ochannels, (1, 1), **self.conv_opts)
            h = self.bn(h, no_relu=True)
        with nn.parameter_scope("se"):
            h = self.seblock(h)
        with nn.parameter_scope("bottleneck_s"):
            s = shortcut(x, ochannels, stride, self.shortcut_type,
                         self.test, channel_last=self.channel_last)
        return F.relu(F.add2(h, s, inplace=True), inplace=True)


class SENet(ResNetBase):
    '''
    A class which defines SE-ResNet or SE-ResNeXt.

    With a ResNet-based architecture, `SEBottleneckBlock` is used for blocks
    in a cell instead of `BottleneckBlock`.

    Args:
        resnext (bool): Defines SE-ResNeXT if True otherwise SE-ResNet.

    See `ResNetBase` for other details.

    '''

    def __init__(self, num_classes=1000, num_layers=50, resnext=False, test=True, channel_last=False):
        assert num_layers == 50, 'num_layers=50 is only supported so far.'
        seblock = SEBottleneckBlock(resnext, 16, 'b', test, channel_last)
        super(SENet, self).__init__(num_classes, num_layers, max_pooling_ceil_border=True,
                                    block=seblock, test=test, channel_last=channel_last)


def se_resnet50(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines SE-ResNet50.

    See `SENet` for more details.

    '''
    net = SENet(num_classes, 50, test=test, channel_last=channel_last)
    return net(x)


def se_resnext50(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines SE-ResNeXt50.

    See `SENet` for more details.

    '''

    net = SENet(num_classes, 50, resnext=True,
                test=test, channel_last=channel_last)
    return net(x)


# Register arch functions
registry.register_arch_fn('se_resnet50', se_resnet50)
registry.register_arch_fn('se_resnext50', se_resnext50)
