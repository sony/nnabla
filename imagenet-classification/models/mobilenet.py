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
import nnabla.parametric_functions as PF
import nnabla.functions as F
import nnabla.initializer as I

import numpy as np
from functools import partial

from .base import (
    pf_convolution,
    get_channel_axis,
    get_spatial_shape,
)
from . import registry


class MobileNetBase(object):

    def __init__(self, num_classes=1000, test=True, depth_mul=1.0, channel_last=False):
        self.num_classes = num_classes
        self.test = test
        self.depth_mul = depth_mul
        self.channel_last = channel_last

        self.act_map = dict(
            relu=partial(F.relu, inplace=True),
            relu6=F.relu6,
            hswish=self.hswish,
            linear=lambda x: x)

    def hswish(self, x):
        return x * (F.relu6(x + 3.0)) / 6.0

    def conv_bn_act(self, x, maps=32, kernel=(3, 3), stride=(1, 1), group=1, act="linear",
                    name="conv-bn"):
        conv_opts = dict(stride=stride, group=group,
                         channel_last=self.channel_last, with_bias=False)
        axes = [get_channel_axis(self.channel_last)]
        with nn.parameter_scope(name):
            h = pf_convolution(x, maps, kernel, **conv_opts)
            h = PF.batch_normalization(h, axes, batch_stat=not self.test)
            h = self.act_map[act](h)
        return h

    def conv_act(self, x, maps=32, kernel=(3, 3), stride=(1, 1), group=1, act="linear",
                 name="conv-bn"):
        conv_opts = dict(stride=stride, group=group,
                         channel_last=self.channel_last, with_bias=False)
        with nn.parameter_scope(name):
            h = pf_convolution(x, maps, kernel, **conv_opts)
            h = self.act_map[act](h)
        return h

    def conv_bn_relu(self, x, maps=32, kernel=(3, 3), stride=(1, 1), group=1, name="conv-bn"):
        h = self.conv_bn_act(x, maps, kernel, stride=stride,
                             group=group, act="relu", name=name)
        return h

    def conv_bn_relu6(self, x, maps=32, kernel=(3, 3), stride=(1, 1), group=1, name="conv-bn"):
        h = self.conv_bn_act(x, maps, kernel, stride=stride,
                             group=group, act="relu6", name=name)
        return h

    def conv_bn(self, x, maps=32, kernel=(3, 3), stride=(1, 1), group=1, name="conv-bn"):
        h = self.conv_bn_act(x, maps, kernel, stride=stride,
                             group=group, act="linear", name=name)
        return h


class MobileNetV1(MobileNetBase):

    def __init__(self, num_classes=1000, test=True, depth_mul=1.0, channel_last=False):
        super(MobileNetV1, self).__init__(
            num_classes, test, depth_mul, channel_last)
        # TODO: where to multiply depth_mul

    def depthwise_separable_conv(self, x, maps=32, stride=(1, 1), name="conv-ds"):
        c = x.shape[get_channel_axis(self.channel_last)]
        with nn.parameter_scope(name):
            h = self.conv_bn_relu(
                x, c, (3, 3), stride=stride, group=c, name="conv-dw")
            h = self.conv_bn_relu(h, maps, (1, 1), stride=(
                1, 1), group=1, name="conv-pw")
        return h

    def __call__(self, x):
        h = self.conv_bn_relu(x, 32, stride=(2, 2), name="first-conv")
        h = self.depthwise_separable_conv(
            h, 64, stride=(1, 1), name="conv-ds-1")
        h = self.depthwise_separable_conv(
            h, 128, stride=(2, 2), name="conv-ds-2")
        h = self.depthwise_separable_conv(
            h, 128, stride=(1, 1), name="conv-ds-3")
        h = self.depthwise_separable_conv(
            h, 256, stride=(2, 2), name="conv-ds-4")
        h = self.depthwise_separable_conv(
            h, 256, stride=(1, 1), name="conv-ds-5")
        h = self.depthwise_separable_conv(
            h, 512, stride=(2, 2), name="conv-ds-6")
        h = self.depthwise_separable_conv(
            h, 512, stride=(1, 1), name="conv-ds-7")
        h = self.depthwise_separable_conv(
            h, 512, stride=(1, 1), name="conv-ds-8")
        h = self.depthwise_separable_conv(
            h, 512, stride=(1, 1), name="conv-ds-9")
        h = self.depthwise_separable_conv(
            h, 512, stride=(1, 1), name="conv-ds-10")
        h = self.depthwise_separable_conv(
            h, 512, stride=(1, 1), name="conv-ds-11")
        h = self.depthwise_separable_conv(
            h, 1024, stride=(2, 2), name="conv-ds-12")
        h = self.depthwise_separable_conv(
            h, 1024, stride=(1, 1), name="conv-ds-13")
        h = F.average_pooling(h, get_spatial_shape(h.shape, self.channel_last))
        h = PF.affine(h, self.num_classes,
                      w_init=I.NormalInitializer(0.01), name="linear")
        return h, {}


class MobileNetV2(MobileNetBase):

    def __init__(self, num_classes=1000, test=True, depth_mul=1.0, channel_last=False):
        super(MobileNetV2, self).__init__(
            num_classes, test, depth_mul, channel_last)

        self.init_maps = 32
        self.settings = [
            # t, c, n, s: expansion factor, maps, num blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

    def inverted_residual(self, x, maps=32, kernel=(3, 3), stride=(1, 1), ef=6,
                          act="relu",
                          name="inv-resblock"):
        h = x
        c = h.shape[get_channel_axis(self.channel_last)]
        hmaps = round(c * ef)
        omaps = maps

        def first_layer(h):
            with nn.parameter_scope(name):
                h = self.conv_bn_relu6(
                    h, hmaps, kernel, stride=stride, group=hmaps, name="conv-dw")
                h = self.conv_bn(h, omaps, (1, 1), stride=(
                    1, 1), name="conv-pw-linear")
            return h

        def other_layer(h):
            h = self.conv_bn_relu6(
                h, hmaps, (1, 1), stride=(1, 1), name="conv-pw")
            h = self.conv_bn_relu6(
                h, hmaps, kernel, stride=stride, group=hmaps, name="conv-dw")
            h = self.conv_bn(h, omaps, (1, 1), stride=(
                1, 1), name="conv-pw-linear")
            return h

        with nn.parameter_scope(name):
            if ef == 1:
                h = first_layer(h)
            else:
                h = other_layer(h)

        use_res_connect = (stride == (1, 1) and c == omaps)
        if use_res_connect:
            h = x + h

        return h

    def __call__(self, x):
        # First conv
        h = self.conv_bn_relu6(x, int(self.init_maps * self.depth_mul),
                               stride=(2, 2), name="first-conv")

        # Inverted residual blocks
        for i, elm in enumerate(self.settings):
            t, c, n, s = elm
            # TODO: where to multiply depth_mul
            c = round(c * self.depth_mul)
            mbconv_s = partial(self.inverted_residual,
                               maps=c, stride=(s, s), ef=t)
            mbconv_1 = partial(self.inverted_residual,
                               maps=c, stride=(1, 1), ef=t)
            for j in range(n):
                name = "mbconv-{:02d}-{:02d}".format(i, j)
                h = mbconv_s(h, name=name) if j == 0 else mbconv_1(
                    h, name=name)
        # Last conv
        h = self.conv_bn_relu6(h, int(1280 * self.depth_mul),
                               kernel=(1, 1), name="last-conv")

        # Classifier
        if not self.test:
            h = F.dropout(h, 0.2)
        h = F.average_pooling(h, get_spatial_shape(h.shape, self.channel_last))
        h = PF.affine(h, self.num_classes,
                      w_init=I.NormalInitializer(0.01), name="linear")

        return h, {}


class MobileNetV3(MobileNetBase):

    def __init__(self, num_classes=1000, test=True, depth_mul=1.0, mode="large", channel_last=False):
        super(MobileNetV3, self).__init__(
            num_classes, test, depth_mul, channel_last)
        self.mode = mode
        if mode not in ["large", "small"]:
            raise ValueError(
                "mode should be in [{}, {}]".format("large", "small"))

        large_settings = [
                          # maps, kernel, stride, expansion factor, activation, se
                          [16, (3, 3), (1, 1), 1, "relu", False],
                          [24, (3, 3), (2, 2), 4, "relu", False],
                          [24, (3, 3), (1, 1), 3, "relu", False],
                          [40, (5, 5), (2, 2), 3, "relu", True],
                          [40, (5, 5), (1, 1), 3, "relu", True],
                          [40, (5, 5), (1, 1), 3, "relu", True],
                          [80, (3, 3), (2, 2), 6, "hswish", False],
                          [80, (3, 3), (1, 1), 2.5, "hswish", False],
                          [80, (3, 3), (1, 1), 2.3, "hswish", False],
                          [80, (3, 3), (1, 1), 2.3, "hswish", False],
                          [112, (3, 3), (1, 1), 6, "hswish", True],
                          [112, (3, 3), (1, 1), 6, "hswish", True],
                          [160, (5, 5), (2, 2), 6, "hswish", True],
                          [160, (5, 5), (1, 1), 6, "hswish", True],
                          [160, (5, 5), (1, 1), 6, "hswish", True]
                          ]
        small_settings = [
                          # maps, kernel, stride, expansion, activation, se
                          [16, (3, 3), (2, 2), 1, "relu", True],
                          [24, (3, 3), (2, 2), 4.5, "relu", False],
                          [24, (3, 3), (1, 1), 3.66, "relu", False],
                          [40, (5, 5), (2, 2), 4, "hswish", True],
                          [40, (5, 5), (1, 1), 6, "hswish", True],
                          [40, (5, 5), (1, 1), 6, "hswish", True],
                          [48, (5, 5), (1, 1), 3, "hswish", True],
                          [48, (5, 5), (1, 1), 3, "hswish", True],
                          [96, (5, 5), (2, 2), 6, "hswish", True],
                          [96, (5, 5), (1, 1), 6, "hswish", True],
                          [96, (5, 5), (1, 1), 6, "hswish", True],
                          ]
        self.maps0 = 16
        self.maps1 = 960 if mode == "large" else 576
        self.maps2 = 1280 if mode == "large" else 1024
        self.settings = large_settings if mode == "large" else small_settings

    def squeeze_and_excite(self, x, rr=4, name="squeeze-and-excite"):
        s = x
        c = x.shape[get_channel_axis(self.channel_last)]
        cr = c // rr
        conv_opts = dict(channel_last=self.channel_last, with_bias=True)

        h = F.average_pooling(x, get_spatial_shape(x.shape, self.channel_last))
        with nn.parameter_scope(name):
            with nn.parameter_scope("fc1"):
                h = pf_convolution(h, cr, (1, 1), **conv_opts)
                h = F.relu(h)
            with nn.parameter_scope("fc2"):
                h = pf_convolution(h, c, (1, 1), **conv_opts)
                h = F.hard_sigmoid(h)
            h = h * s
        return h

    def inverted_residual(self, x, maps=32, kernel=(3, 3), stride=(1, 1), ef=6,
                          act="relu", se=False,
                          name="inv-resblock"):
        h = x
        c = h.shape[get_channel_axis(self.channel_last)]
        hmaps = round(c * ef)
        omaps = maps
        with nn.parameter_scope(name):
            h = self.conv_bn_act(h, hmaps, (1, 1), (1, 1),
                                 act=act, name="conv-pw")
            h = self.conv_bn_act(h, hmaps, kernel, stride,
                                 group=hmaps, act=act, name="conv-dw")
            h = self.squeeze_and_excite(
                h, name="squeeze-and-excite") if se else h
            h = self.conv_bn(h, omaps, (1, 1), stride=(
                1, 1), name="conv-pw-linear")

        use_res_connect = (stride == (1, 1) and c == omaps)
        if use_res_connect:
            h = x + h
        return h

    def __call__(self, x):
        # First conv
        h = self.conv_bn_act(x, int(self.maps0 * self.depth_mul),
                             stride=(2, 2), act="hswish", name="first-conv")

        # Inverted residual blocks
        for i, elm in enumerate(self.settings):
            maps, kernel, stride, ef, act, se = elm
            maps = round(maps * self.depth_mul)
            name = "mbconv-{:03d}".format(i)
            h = self.inverted_residual(
                h, maps, kernel, stride, ef, act, se, name=name)

        # Conv -> Avepool -> Conv
        h = self.conv_bn_act(h, int(self.maps1 * self.depth_mul), (1, 1), act="hswish",
                             name="last-conv-1")
        h = F.average_pooling(h, get_spatial_shape(h.shape, self.channel_last))
        h = self.conv_act(h, int(self.maps2 * self.depth_mul), (1, 1), act="hswish",
                          name="last-conv-2")

        # Classifier
        if not self.test:
            h = F.dropout(h, 0.2)
        h = PF.affine(h, self.num_classes,
                      w_init=I.NormalInitializer(0.01), name="linear")

        return h, {}


def mobilenet_v1(x, num_classes=1000, test=True, depth_mul=1.0, channel_last=False):
    net = MobileNetV1(num_classes, test, depth_mul, channel_last)
    return net(x)


def mobilenet_v2(x, num_classes=1000, test=True, depth_mul=1.0, channel_last=False):
    net = MobileNetV2(num_classes, test, depth_mul, channel_last)
    return net(x)


def mobilenet_v3_large(x, num_classes=1000, test=True, depth_mul=1.0, channel_last=False):
    net = MobileNetV3(num_classes, test, depth_mul, "large", channel_last)
    return net(x)


def mobilenet_v3_small(x, num_classes=1000, test=True, depth_mul=1.0, channel_last=False):
    net = MobileNetV3(num_classes, test, depth_mul, "small", channel_last)
    return net(x)


# Register arch functions
for v in ["v1", "v2", "v3_large", "v3_small"]:
    fn_name = 'mobilenet_{}'.format(v)
    registry.register_arch_fn(fn_name, eval(fn_name))
