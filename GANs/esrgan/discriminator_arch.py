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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def conv_bn_3(x, nf, name, bias):
    with nn.parameter_scope(name):
        h = PF.convolution(x, nf, kernel=(3, 3), stride=(
            1, 1), pad=(1, 1), with_bias=bias)
        h = PF.batch_normalization(h)
        h = F.leaky_relu(h, alpha=0.2, inplace=True)
    return h


def conv_bn_4(x, nf, name, bias):
    with nn.parameter_scope(name):
        h = PF.convolution(x, nf, kernel=(4, 4), stride=(
            2, 2), pad=(1, 1), with_bias=bias)
        h = PF.batch_normalization(h)
        h = F.leaky_relu(h, alpha=0.2, inplace=True)
    return h


def discriminator(x, nf=64):
    '''
    :param x: input to the network
    :param nf: number of output channels
    :return:
    '''
    # [3,128, 128]
    h = F.leaky_relu(PF.convolution(x, nf, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name='conv0_0', with_bias=True),
                     alpha=0.2, inplace=True)
    # [64, 128, 128]
    h = conv_bn_4(h, nf, "conv0_1", False)
    h = conv_bn_3(h, 2 * nf, "conv1_0", False)
    h = conv_bn_4(h, 2 * nf, "conv1_1", False)
    h = conv_bn_3(h, 4 * nf, "conv2_0", False)
    h = conv_bn_4(h, 4 * nf, "conv2_1", False)
    h = conv_bn_3(h, 8 * nf, "conv3_0", False)
    h = conv_bn_4(h, 8 * nf, "conv3_1", False)
    h = conv_bn_3(h, 8 * nf, "conv4_0", False)
    h = conv_bn_4(h, 8 * nf, "conv4_1", False)
    # [512, 4, 4]
    B, C, H, W = h.shape[0], h.shape[1], h.shape[2], h.shape[3]
    h = F.leaky_relu((PF.affine(h, 100, name="affine1")),
                     alpha=0.2, inplace=True)
    h = PF.affine(h, 1, name="affine2")
    return h
