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

import numpy as np

import nnabla as nn
import nnabla.functions as F

from ops import upsample_conv_2d, weight_init_fn


def mapping_network(noise, outmaps=512, num_layers=8, net_scope='G_mapping/Dense'):
    """
        a mapping network which embeds input noise into a vector in latent space.
        activation layer contains multiplication by np.sqrt(2).
    """
    lrmul = 0.01
    runtime_coef = 0.00044194172 
        
    out = noise
    for i in range(num_layers):
        with nn.parameter_scope(f'{net_scope}{i}'):
            W, bias = weight_init_fn(shape=(out.shape[1], outmaps), lrmul=lrmul)
            out = F.affine(out, W*runtime_coef, bias*lrmul)
            out = F.mul_scalar(F.leaky_relu(out, alpha=0.2, inplace=True), np.sqrt(2), inplace=True)
    return out

def conv_block(input, w, noise=None, res=4, outmaps=512, inmaps=512,
               kernel_size=3, pad_size=1, demodulate=True, namescope="Conv",
               up=False, act=F.leaky_relu):
    """
        single convoluiton block used in each resolution.
    """

    batch_size = input.shape[0]
    with nn.parameter_scope(f"G_synthesis/{res}x{res}/{namescope}"):
        runtime_coef = 1. / np.sqrt(512)
        W, bias = weight_init_fn(shape=(w.shape[1], inmaps)) 
        runtime_coef = 1. / np.sqrt(512) 
        s = F.affine(w, W*runtime_coef, bias) + 1.0
    runtime_coef_for_conv = 1/np.sqrt(np.prod([inmaps, kernel_size, kernel_size]))

    runtime_coef_for_conv = 1 / \
        np.sqrt(np.prod([inmaps, kernel_size, kernel_size]))

    if up:
        conv_weight = nn.parameter.get_parameter_or_create(name=f"G_synthesis/{res}x{res}/{namescope}/conv/W", shape=(inmaps, outmaps, kernel_size, kernel_size))
    else:
        conv_weight = nn.parameter.get_parameter_or_create(name=f"G_synthesis/{res}x{res}/{namescope}/conv/W", shape=(outmaps, inmaps, kernel_size, kernel_size))
    conv_weight = conv_weight * runtime_coef_for_conv

    if up:
        scale = F.reshape(s, (s.shape[0], s.shape[1], 1, 1, 1), inplace=True)
    else:
        scale = F.reshape(s, (s.shape[0], 1, s.shape[1], 1, 1), inplace=True)

    mod_w = F.mul2(F.reshape(conv_weight, (1,) +
                             conv_weight.shape, inplace=True), scale)

    if demodulate:
        if up:
            denom_w = F.pow_scalar(F.sum(F.pow_scalar(mod_w, 2.), axis=[
                                   1, 3, 4], keepdims=True) + 1e-8, 0.5)
        else:
            denom_w = F.pow_scalar(F.sum(F.pow_scalar(mod_w, 2.), axis=[
                                   2, 3, 4], keepdims=True) + 1e-8, 0.5)
        demod_w = F.div2(mod_w, denom_w)

    else:
        demod_w = mod_w

    input = F.reshape(
        input, (1, -1, input.shape[2], input.shape[3]), inplace=True)
    demod_w = F.reshape(
        demod_w, (-1, demod_w.shape[2], demod_w.shape[3], demod_w.shape[4]), inplace=True)

    if up:
        k = [1, 3, 3, 1]
        conv_out = upsample_conv_2d(
            input, demod_w, k, factor=2, gain=1, group=batch_size)
    else:
        conv_out = F.convolution(input, demod_w, pad=(
            pad_size, pad_size), group=batch_size)
        conv_out = F.reshape(
            conv_out, (batch_size, -1, conv_out.shape[2], conv_out.shape[3]), inplace=True)

    if noise is not None:
        noise_coeff = nn.parameter.get_parameter_or_create(name=f"G_synthesis/{res}x{res}/{namescope}/noise_strength", shape=())
        output = conv_out + noise * \
            F.reshape(noise_coeff, (1, 1, 1, 1), inplace=False)
    else:
        output = conv_out

    bias = nn.parameter.get_parameter_or_create(name=f"G_synthesis/{res}x{res}/{namescope}/conv/b", shape=(outmaps,))
    output = output + F.reshape(bias, (1, outmaps, 1, 1), inplace=False)

    if act == F.leaky_relu:
        output = F.leaky_relu(output, alpha=0.2) * np.sqrt(2)
    else:
        output = act(output)
    return output
