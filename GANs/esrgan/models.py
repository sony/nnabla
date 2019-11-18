# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
import functools
import numpy as np


def residual_block_5C(x, num_output_channel=64, growth_channel=32):
    conv1 = F.leaky_relu(PF.convolution(x, growth_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='conv1'), alpha=0.2, inplace=True)
    conv2 = F.leaky_relu(PF.convolution(F.concatenate(x, conv1, axis=1), growth_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='conv2'), alpha=0.2, inplace=True)
    conv3 = F.leaky_relu(PF.convolution(F.concatenate(x, conv1, conv2, axis=1), growth_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='conv3'), alpha=0.2, inplace=True)
    conv4 = F.leaky_relu(PF.convolution(F.concatenate(x, conv1, conv2, conv3, axis=1), growth_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='conv4'), alpha=0.2, inplace=True)
    conv5 = PF.convolution(F.concatenate(x, conv1, conv2, conv3, conv4, axis=1),
                           num_output_channel, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv5')
    return (conv5 * 0.2) + x


def rrdb(x, num_output_channel=64, growth_channel=32):
    with nn.parameter_scope('RDB1'):
        rdb1 = residual_block_5C(x, num_output_channel, growth_channel)
    with nn.parameter_scope('RDB2'):
        rdb2 = residual_block_5C(rdb1, num_output_channel, growth_channel)
    with nn.parameter_scope('RDB3'):
        rdb3 = residual_block_5C(rdb2, num_output_channel, growth_channel)

    return (rdb3 * 0.2) + x


def rrdb_net(x, num_output_channel, num_rrdb_blocks, growth_channel=32):
    '''
    :param x: input image
    :param num_output_channel: number of output channels
    :param num_rrdb_blocks: number of residual blocks
    :param growth_channel: growth channel (no. of intermediate channel)
    :return:
    '''
    fea = PF.convolution(x, num_output_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='conv_first')
    h = fea
    with nn.parameter_scope('RRDB_trunk'):
        for i in range(num_rrdb_blocks):
            with nn.parameter_scope('{}'.format(i)):
                h = rrdb(h, num_output_channel, growth_channel)

    trunk_conv = PF.convolution(h, num_output_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='trunk_conv')
    fea = fea + trunk_conv
    up_conv1 = F.leaky_relu(PF.convolution(F.interpolate(fea, scale=(2, 2), mode='nearest'), num_output_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='upconv1'), alpha=0.2, inplace=True)
    up_conv2 = F.leaky_relu(PF.convolution(F.interpolate(up_conv1, scale=(2, 2), mode='nearest'), num_output_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='upconv2'), alpha=0.2, inplace=True)
    hr_conv = F.leaky_relu(PF.convolution(up_conv2, num_output_channel, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='HRconv'), alpha=0.2, inplace=True)
    conv_last = PF.convolution(hr_conv, 3, kernel=(
        3, 3), stride=(1, 1), pad=(1, 1), name='conv_last')
    return conv_last
