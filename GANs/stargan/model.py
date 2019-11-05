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
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


def resblock(x, dim_out, w_init=None, epsilon=1e-05):
    assert dim_out == x.shape[1], "The number of input / output channels must match."
    h = PF.convolution(x, dim_out, kernel=(3, 3), pad=(
        1, 1), with_bias=False, w_init=w_init, name="1st")
    h = PF.instance_normalization(h, eps=epsilon, name="1st")
    h = F.relu(h, inplace=True)
    h = PF.convolution(h, dim_out, kernel=(3, 3), pad=(
        1, 1), with_bias=False, w_init=w_init, name="2nd")
    h = PF.instance_normalization(h, eps=epsilon, name="2nd")
    return x + h


def generator(x, c, conv_dim=64, c_dim=5, num_downsample=2, num_upsample=2, repeat_num=6, w_init=None, epsilon=1e-05):
    assert len(c.shape) == 4
    c = F.tile(c, (1, 1) + x.shape[2:])
    concat_input = F.concatenate(x, c, axis=1)

    h = PF.convolution(concat_input, conv_dim, kernel=(7, 7), pad=(
        3, 3), stride=(1, 1), with_bias=False, w_init=w_init, name="init_conv")
    h = PF.instance_normalization(h, eps=epsilon, name="init_inst_norm")
    h = F.relu(h, inplace=True)

    # Down-sampling layers.
    curr_dim = conv_dim
    for i in range(num_downsample):
        h = PF.convolution(h, curr_dim*2, kernel=(4, 4), pad=(1, 1), stride=(2, 2),
                           with_bias=False, w_init=w_init, name="downsample_{}".format(i))
        h = PF.instance_normalization(
            h, eps=epsilon, name="downsample_{}".format(i))
        h = F.relu(h, inplace=True)
        curr_dim = curr_dim * 2

    # Bottleneck layers.
    for i in range(repeat_num):
        with nn.parameter_scope("bottleneck_{}".format(i)):
            h = resblock(h, dim_out=curr_dim)

    # Up-sampling layers.
    for i in range(num_upsample):
        h = PF.deconvolution(h, curr_dim//2, kernel=(4, 4), pad=(1, 1), stride=(
            2, 2), w_init=w_init, with_bias=False, name="upsample_{}".format(i))
        h = PF.instance_normalization(
            h, eps=epsilon, name="upsample_{}".format(i))
        h = F.relu(h, inplace=True)
        curr_dim = curr_dim // 2

    h = PF.convolution(h, 3, kernel=(7, 7), pad=(3, 3), stride=(
        1, 1), with_bias=False, w_init=w_init, name="last_conv")
    h = F.tanh(h)
    return h


def discriminator(x, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, w_init=None):
    assert x.shape[-1] == image_size, "image_size and input spatial size must match."
    h = PF.convolution(x, conv_dim, kernel=(4, 4), pad=(
        1, 1), stride=(2, 2), w_init=w_init, name="init_conv")
    h = F.leaky_relu(h, alpha=0.01)

    curr_dim = conv_dim
    for i in range(1, repeat_num):
        h = PF.convolution(h, curr_dim*2, kernel=(4, 4), pad=(1, 1),
                           stride=(2, 2), w_init=w_init, name="downsample_{}".format(i))
        h = F.leaky_relu(h, alpha=0.01)
        curr_dim = curr_dim * 2

    kernel_size = int(image_size / np.power(2, repeat_num))

    out_src = PF.convolution(h, 1, kernel=(3, 3), pad=(1, 1), stride=(
        1, 1), with_bias=False, w_init=w_init, name="last_1x1_conv")
    out_cls = PF.convolution(h, c_dim, kernel=(
        kernel_size, kernel_size), with_bias=False, w_init=w_init, name="last_cls_conv")
    return out_src, out_cls
