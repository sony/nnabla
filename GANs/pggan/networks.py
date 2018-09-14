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


import functools
from nnabla import Variable
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.parametric_functions import parametric_function_api
import os
import sys

from functions import (minibatch_stddev,
                       BN,
                       LN,
                       pixel_wise_feature_vector_normalization,
                       conv,
                       affine)
import nnabla as nn
import nnabla.functions as F
import nnabla.initializer as I
import nnabla.parametric_functions as PF
import numpy as np


class Generator:
    def __init__(self, use_bn=False, last_act='tanh',
                 use_wscale=True, use_he_backward=False):
        self.resolution_list = []
        self.channel_list = []

        self.activation = functools.partial(F.relu, inplace=True)
        if last_act == 'tanh':
            self.last_act = F.tanh
        else:
            logger.info("Set the last activation function of the generator.")
            sys.exit(0)
        self.use_bn = use_bn
        self.use_wscale = use_wscale
        self.use_he_backward = use_he_backward

    def __call__(self, x, test=False):
        """Generate images.
        """
        with nn.parameter_scope("generator"):
            h = self.first_cnn(
                x, self.resolution_list[0], self.channel_list[0], test)
            for i in range(1, len(self.resolution_list)):
                h = self.cnn(
                    h, self.resolution_list[i], self.channel_list[i], test)
            h = self.to_RGB(h, self.resolution_list[-1])
            h = self.last_act(h)
        return h

    def transition(self, x, alpha, test=False):
        """Generator in the transition period, almost the same as the callable.
        """
        with nn.parameter_scope("generator"):
            h = self.first_cnn(
                x, self.resolution_list[0], self.channel_list[0], test)
            for i in range(1, len(self.resolution_list) - 1):
                h = self.cnn(
                    h, self.resolution_list[i], self.channel_list[i], test)
            h = self.transition_cnn(h, self.resolution_list[-2], self.resolution_list[-1],
                                    self.channel_list[-2], self.channel_list[-1], alpha, test)
            h = self.last_act(h)
        return h

    def grow(self, resolution, channel):
        """Add resolution and channel.
        """
        self.resolution_list.append(resolution)
        self.channel_list.append(channel)

    def to_RGB(self, h, resolution):
        """To RGB layer

        To RGB projects feature maps to RGB maps.
        """
        with nn.parameter_scope("to_rgb_{}".format(resolution)):
            h = conv(h, 3, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                     with_bias=True,
                     use_wscale=self.use_wscale,
                     use_he_backward=self.use_he_backward)
        return h

    def first_cnn(self, h, resolution, channel, test=False):
        with nn.parameter_scope("phase_{}".format(resolution)):
            # affine is 1x1 conv with 4x4 kernel and 3x3 pad.
            with nn.parameter_scope("conv1"):
                h = affine(h, channel * 4 * 4, with_bias=not self.use_bn,
                           use_wscale=self.use_wscale,
                           use_he_backward=self.use_he_backward)
                h = BN(h, use_bn=self.use_bn, test=test)
                h = F.reshape(h, (h.shape[0], channel, 4, 4))
                h = pixel_wise_feature_vector_normalization(
                    BN(h, use_bn=self.use_bn, test=test))
                h = self.activation(h)
            with nn.parameter_scope("conv2"):
                h = conv(h, channel, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         with_bias=not self.use_bn,
                         use_wscale=self.use_wscale,
                         use_he_backward=self.use_he_backward)
                h = pixel_wise_feature_vector_normalization(
                    BN(h, use_bn=self.use_bn, test=test))
                h = self.activation(h)
        return h

    def cnn(self, h, resolution, channel, test):
        """CNN block

        The following operations are performed two times.

        1. Upsampling
        2. Conv
        3. Pixel-wise normalization
        4. Relu
        """
        h = F.unpooling(h, kernel=(2, 2))
        with nn.parameter_scope("phase_{}".format(resolution)):
            with nn.parameter_scope("conv1"):
                h = conv(h, channel, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         with_bias=not self.use_bn,
                         use_wscale=self.use_wscale,
                         use_he_backward=self.use_he_backward)
                h = pixel_wise_feature_vector_normalization(
                    BN(h, use_bn=self.use_bn, test=test))
                h = self.activation(h)
            with nn.parameter_scope("conv2"):
                h = conv(h, channel, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         with_bias=not self.use_bn,
                         use_wscale=self.use_wscale,
                         use_he_backward=self.use_he_backward)
                h = pixel_wise_feature_vector_normalization(
                    BN(h, use_bn=self.use_bn, test=test))
                h = self.activation(h)
        return h

    def transition_cnn(self, h, pre_resolution, nxt_resolution, pre_channel, nxt_channel, alpha, test):
        lhs = self.to_RGB(F.unpooling(h, kernel=(2, 2)), pre_resolution)
        rhs = self.to_RGB(self.cnn(h, nxt_resolution,
                                   nxt_channel, test), nxt_resolution)
        return (1 - alpha) * lhs + alpha * rhs

    def save_parameters(self, monitor_path, file_name):
        path_params = "{}/{}.h5".format(monitor_path, file_name)
        with nn.parameter_scope("generator"):
            nn.save_parameters(path_params)


class Discriminator:
    def __init__(self, use_ln=False, alpha=0.1,
                 use_wscale=True, use_he_backward=False):
        self.resolution_list = []
        self.channel_list = []

        # TODO: alpha = 0.1?
        self.activation = functools.partial(F.leaky_relu, alpha=alpha)
        self.use_ln = use_ln
        self.use_wscale = use_wscale
        self.use_he_backward = use_he_backward

    def __call__(self, x):
        """Discriminate images.
        """
        with nn.parameter_scope("discriminator"):
            h = self.from_RGB(
                x, self.resolution_list[-1], self.channel_list[-1])
            for i in range(1, len(self.resolution_list))[::-1]:
                h = self.cnn(h, self.resolution_list[i],
                             self.channel_list[i], self.channel_list[i - 1])
            h = minibatch_stddev(h)
            h = self.last_cnn(h, self.resolution_list[0], self.resolution_list[0],
                              self.resolution_list[0])
        return h

    def transition(self, x, alpha):
        """Generator in the transition period, almost the same as the callable.

        """
        with nn.parameter_scope("discriminator"):
            h = self.transition_cnn(x, self.resolution_list[-2], self.resolution_list[-1],
                                    self.channel_list[-2], self.channel_list[-1], alpha)
            for i in range(1, len(self.resolution_list) - 1)[::-1]:
                h = self.cnn(h, self.resolution_list[i], self.channel_list[i],
                             self.channel_list[i - 1])
            h = minibatch_stddev(h)
            h = self.last_cnn(h, self.resolution_list[0], self.resolution_list[0],
                              self.resolution_list[0])
        return h

    def grow(self, resolution, channel):
        """Add resolution and channel.
        """
        self.resolution_list.append(resolution)
        self.channel_list.append(channel)

    def from_RGB(self, h, resolution, channel):
        """From RGB layer

        From RGB projects RGB maps to feature maps.
        """
        with nn.parameter_scope("from_rgb_{}".format(resolution)):
            h = conv(h, channel, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                     with_bias=True,
                     use_wscale=self.use_wscale,
                     use_he_backward=self.use_he_backward)
            h = self.activation(h)
        return h

    def cnn(self, h, resolution, ch_in, ch_out):
        """CNN block

        The following operations are performed two times.

        1. Conv
        2. Layer normalization
        3. Leaky relu
        """
        with nn.parameter_scope("phase_{}".format(resolution)):
            with nn.parameter_scope("conv1"):
                h = conv(h, ch_in, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         with_bias=not self.use_ln,
                         use_wscale=self.use_wscale,
                         use_he_backward=self.use_he_backward)
                h = LN(h, use_ln=self.use_ln)
                h = self.activation(h)
            with nn.parameter_scope("conv2"):
                h = conv(h, ch_out, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         with_bias=not self.use_ln,
                         use_wscale=self.use_wscale,
                         use_he_backward=self.use_he_backward)
                h = LN(h, use_ln=self.use_ln)
                h = self.activation(h)
        h = F.average_pooling(h, kernel=(2, 2))
        return h

    def last_cnn(self, h, resolution, ch_in, ch_out):
        with nn.parameter_scope("phase_{}".format(resolution)):
            with nn.parameter_scope("conv1"):
                h = conv(h, ch_in, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         with_bias=not self.use_ln,
                         use_wscale=self.use_wscale,
                         use_he_backward=self.use_he_backward)
                h = LN(h, use_ln=self.use_ln)
                h = self.activation(h)
            with nn.parameter_scope("conv2"):
                h = conv(h, ch_out, kernel=(4, 4), pad=(0, 0), stride=(1, 1),
                         with_bias=not self.use_ln,
                         use_wscale=self.use_wscale,
                         use_he_backward=self.use_he_backward)
                h = LN(h, use_ln=self.use_ln)
                h = self.activation(h)
            with nn.parameter_scope("linear"):
                h = affine(h, 1, with_bias=True,
                           use_wscale=self.use_wscale,
                           use_he_backward=self.use_he_backward)
        return h

    def transition_cnn(self, h, pre_resolution, nxt_resolution, pre_channel, nxt_channel, alpha):
        lhs = self.from_RGB(F.average_pooling(
            h, kernel=(2, 2)), pre_resolution, pre_channel)
        rhs = alpha * self.cnn(self.from_RGB(h, nxt_resolution, nxt_channel),
                               nxt_resolution, nxt_channel, pre_channel)
        return (1 - alpha) * lhs + alpha * rhs

    def save_parameters(self, monitor_path, file_name):
        path_params = "{}/{}.h5".format(monitor_path, file_name)
        with nn.parameter_scope("discriminator"):
            nn.save_parameters(path_params)
