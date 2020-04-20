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
from .mobilenet import MobileNetBase


class EfficientNet(MobileNetBase):
    # TODO: We did not do any trainings/tests for EfficientNet yet.
    # TODO: Parameter count is almost same as the original

    def __init__(self, num_classes=1000, test=True, channel_last=False, mode="b0"):
        super(EfficientNet, self).__init__(
            num_classes, test, channel_last=channel_last)

        self.num_classes = num_classes
        self.test = test
        self.channel_last = channel_last

        self.act_map.update(dict(swish=F.swish))

        self.mbc_settings = [
            # t, c, k, n, s: expansion factor, maps, kernel, num blocks, stride
            [1, 16, 3, 1, 1],
            [6, 24, 3, 2, 2],
            [6, 40, 5, 2, 2],
            [6, 80, 3, 3, 2],
            [6, 112, 5, 3, 1],
            [6, 192, 5, 4, 2],
            [6, 320, 3, 1, 1]
        ]

        self.net_settings = {
            "b0": dict(width_coef=1.0, depth_coef=1.0, resolution=224, p=0.2),
            "b1": dict(width_coef=1.0, depth_coef=1.1, resolution=240, p=0.2),
            "b2": dict(width_coef=1.1, depth_coef=1.2, resolution=260, p=0.3),
            "b3": dict(width_coef=1.2, depth_coef=1.4, resolution=300, p=0.3),
            "b4": dict(width_coef=1.4, depth_coef=1.8, resolution=380, p=0.4),
            "b5": dict(width_coef=1.6, depth_coef=2.2, resolution=456, p=0.4),
            "b6": dict(width_coef=1.8, depth_coef=2.6, resolution=528, p=0.5),
            "b7": dict(width_coef=2.0, depth_coef=3.1, resolution=600, p=0.5),
            "b8": dict(width_coef=2.2, depth_coef=3.6, resolution=672, p=0.5),
            "l2": dict(width_coef=4.3, depth_coef=5.3, resolution=800, p=0.5),
        }

        self.mode = mode
        self.net_setting = self.net_settings[mode]

    def conv_bn_swish(self, x, maps=32, kernel=(3, 3), stride=(1, 1), group=1, name="conv-bn"):
        h = self.conv_bn_act(x, maps, kernel, stride=stride,
                             group=group, act="swish", name=name)
        return h

    def squeeze_and_excite(self, x, rmaps, name="squeeze-and-excite"):
        s = x
        c = x.shape[get_channel_axis(self.channel_last)]
        conv_opts = dict(channel_last=self.channel_last, with_bias=True)

        h = F.average_pooling(x, get_spatial_shape(x.shape, self.channel_last))
        with nn.parameter_scope(name):
            with nn.parameter_scope("fc1"):
                h = pf_convolution(h, rmaps, (1, 1), **conv_opts)
                h = F.relu(h)
            with nn.parameter_scope("fc2"):
                h = pf_convolution(h, c, (1, 1), **conv_opts)
                h = F.sigmoid(h)
            h = h * s
        return h

    def inverted_residual(self, x, maps=32, kernel=(3, 3), stride=(1, 1), ef=6,
                          name="inv-resblock"):
        h = x
        c = h.shape[get_channel_axis(self.channel_last)]
        hmaps = round(c * ef)
        rmaps = c // 4
        omaps = maps

        with nn.parameter_scope(name):
            h = self.conv_bn_swish(h, hmaps, (1, 1), (1, 1), name="conv-pw")
            h = self.conv_bn_swish(
                h, hmaps, kernel, stride, group=hmaps, name="conv-dw")
            h = self.squeeze_and_excite(h, rmaps, name="squeeze-and-excite")
            h = self.conv_bn(h, omaps, (1, 1), stride=(
                1, 1), name="conv-pw-linear")

        use_res_connect = (stride == (1, 1) and c == omaps)
        if use_res_connect:
            h = self.drop_connect(h)
            h = x + h
        return h

    def drop_connect(self, h, p=0.2):
        if self.test:
            return h
        keep_prob = 1.0 - p
        shape = [1 if i != 0 else h.shape[0] for i in range(h.ndim)]
        r = F.rand(shape=shape)
        r += keep_prob
        m = F.floor(r)
        h = h * (m / keep_prob)
        return h

    def round_filters(self, filters, width_coefficient, depth_divisor=8):
        """Round number of filters based on depth multiplier."""
        orig_f = filters
        multiplier = width_coefficient
        divisor = depth_divisor
        filters *= multiplier
        min_depth = divisor
        new_filters = max(min_depth, int(
            filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def __call__(self, x):
        depth_coef = self.net_setting["depth_coef"]
        width_coef = self.net_setting["width_coef"]
        resolution = self.net_setting["resolution"]
        p = self.net_setting["p"]
        if not self.test:
            assert get_spatial_shape(x.shape, self.channel_last) == [resolution, resolution], \
              "(x.shape = {}, resolution = {})".format(x.shape, resolution)

        # First conv
        maps = self.round_filters(32, width_coef)
        h = self.conv_bn(x, maps, stride=(2, 2), name="first-conv")

        # Inverted residual blocks
        for i, elm in enumerate(self.mbc_settings):
            t, c, k, n, s = elm
            c = self.round_filters(c, width_coef)
            n = int(np.ceil(n * depth_coef))
            mbconv_s = partial(self.inverted_residual, maps=c,
                               kernel=(k, k), stride=(s, s), ef=t)
            mbconv_1 = partial(self.inverted_residual, maps=c,
                               kernel=(k, k), stride=(1, 1), ef=t)
            for j in range(n):
                name = "mbconv-{:02d}-{:02d}".format(i, j)
                h = mbconv_s(h, name=name) if j == 0 else mbconv_1(
                    h, name=name)
        # Last conv
        maps = self.round_filters(1280, width_coef)
        h = self.conv_bn_swish(h, maps, kernel=(1, 1), name="last-conv")

        # Classifier
        if not self.test:
            h = F.dropout(h, p)
        h = F.average_pooling(h, get_spatial_shape(h.shape, self.channel_last))
        h = PF.affine(h, self.num_classes,
                      w_init=I.NormalInitializer(0.01), name="linear")

        return h, {}


def efficientnet_b0(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b0")
    return net(x)


def efficientnet_b1(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b1")
    return net(x)


def efficientnet_b2(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b2")
    return net(x)


def efficientnet_b3(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b3")
    return net(x)


def efficientnet_b4(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b4")
    return net(x)


def efficientnet_b5(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b5")
    return net(x)


def efficientnet_b6(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b6")
    return net(x)


def efficientnet_b7(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b7")
    return net(x)


def efficientnet_b8(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "b8")
    return net(x)


def efficientnet_l2(x, num_classes=1000, test=True, channel_last=False):
    net = EfficientNet(num_classes, test, channel_last, "l2")
    return net(x)


# Register arch functions
for mode in ["b{}".format(i) for i in range(9)] + ["l2"]:
    fn_name = 'efficientnet_{}'.format(mode)
    registry.register_arch_fn(fn_name, eval(fn_name))
