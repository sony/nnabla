# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.core.module import Module  # TODO


def complete_dims(s, dims):
    if not hasattr(s, '__iter__'):
        return (s,) * dims
    if len(s) == dims:
        return s
    raise ValueError('')


def get_conv_same_pad(k):
    return tuple(kk // 2 for kk in k)


class ConvBn(Module):
    def __init__(self, outmaps, kernel=1, stride=1, act=None):
        self.outmaps = outmaps
        self.kernel = kernel
        self.stride = stride
        self.act = act

    def call(self, x, training=True):
        kernel = complete_dims(self.kernel, 2)
        pad = get_conv_same_pad(kernel)
        stride = complete_dims(self.stride, 2)
        h = PF.convolution(x, self.outmaps, kernel,
                           pad, stride, with_bias=False)
        h = PF.batch_normalization(h, batch_stat=training)
        if self.act is None:
            return h
        return self.act(h)


class ResUnit(Module):
    def __init__(self, channels, stride=1, skip_by_conv=True):
        self.conv1 = ConvBn(channels // 4, 1, 1,
                            act=lambda x: F.relu(x, inplace=True))
        self.conv2 = ConvBn(channels // 4, 3, stride,
                            act=lambda x: F.relu(x, inplace=True))
        self.conv3 = ConvBn(channels, 1)
        self.skip_by_conv = skip_by_conv
        self.skip = ConvBn(channels, 1, stride)

    def call(self, x, training=True):

        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        s = x
        if self.skip_by_conv:
            s = self.skip(s)
        h = F.relu(F.add2(h, s, inplace=True), inplace=True)
        return h
