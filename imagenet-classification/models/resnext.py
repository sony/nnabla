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
    shortcut,
)
from . import registry


class ResNeXtBottleneckBlock(BottleneckBlock):
    '''
    Bottleneck block for ResNeXt

    Args:
        cardinality (int):
            Specifies group size of the second convolution at bottleneck.
            The default is 32 which is used in the paper.

    See `BottleneckBlock` for other details.

    '''

    def __init__(self, cardinality=32, shortcut_type='b', test=True, channel_last=False):
        super(ResNeXtBottleneckBlock, self).__init__(
            shortcut_type, test, channel_last)
        self.cardinality = cardinality

    def __call__(self, x, ochannels, stride):
        div = 2
        assert ochannels % div == 0
        hchannels = ochannels // div
        with nn.parameter_scope("bottleneck1"):
            h = self.bn(pf_convolution(x, hchannels, (1, 1), **self.conv_opts))
        with nn.parameter_scope("bottleneck2"):
            h = self.bn(pf_convolution(h, hchannels, (3, 3),
                                       stride=stride,
                                       group=self.cardinality,
                                       **self.conv_opts))
        with nn.parameter_scope("bottleneck3"):
            h = pf_convolution(h, ochannels, (1, 1), **self.conv_opts)
        with nn.parameter_scope("bottleneck_s"):
            s = shortcut(x, ochannels, stride, self.shortcut_type,
                         self.test, channel_last=self.channel_last)
        with nn.parameter_scope("bottleneck3"):  # backward compat.
            h = self.bn(h, s)
        return h


class ResNeXt(ResNetBase):
    '''
    A class which defines ResNeXt.

    With a ResNet-based architecture, `SEBottleneckBlock` is used for blocks
    in a cell instead of `BottleneckBlock`.

    Args:
        cardinality (int): See `ResNeXtBottleneckBlock`.

    See `ResNetBase` for other details.

    '''

    def __init__(self, num_classes=1000, num_layers=50, cardinality=32,
                 test=True, channel_last=False):
        block = ResNeXtBottleneckBlock(cardinality, 'b', test, channel_last)
        super(ResNeXt, self).__init__(
            num_classes, num_layers, max_pooling_ceil_border=True,
            block=block, test=test, channel_last=channel_last)


def resnext50(x, num_classes=1000, test=True, channel_last=False):
    '''
    Defines ResNeXt50.

    See `ResNeXt` for more details.

    '''
    net = ResNeXt(num_classes, 50, test=test, channel_last=channel_last)
    return net(x)


# Register arch functions
registry.register_arch_fn('resnext50', resnext50)
