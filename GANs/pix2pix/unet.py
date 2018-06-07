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

# nnabla imports
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def conv_bn_relu(inp, maps, size, stride=(1, 1), pad=(0, 0),
                 deconv=False, bn=True, dropout=False, relu=F.relu,
                 test=False, name=''):
    if not deconv:
        if isinstance(inp, tuple):
            h = F.add2(
                PF.convolution(inp[0], maps, size, stride=stride,
                               pad=pad, with_bias=not bn, name=name+'_conv_0'),
                PF.convolution(inp[1], maps, size, stride=stride,
                               pad=pad, with_bias=False, name=name+'_conv_1'),
                inplace=True)
        else:
            h = PF.convolution(inp, maps, size, stride=stride,
                               pad=pad, with_bias=not bn, name=name+'_conv')
    else:
        if isinstance(inp, tuple):
            h = F.add2(
                PF.deconvolution(inp[0], maps, kernel=size, stride=stride,
                                 pad=pad, with_bias=not bn, name=name+'_deconv_0'),
                PF.deconvolution(inp[1], maps, kernel=size, stride=stride,
                                 pad=pad, with_bias=False, name=name+'_deconv_1'),
                inplace=True)
        else:
            h = PF.deconvolution(inp, maps, kernel=size, stride=stride,
                                 pad=pad, with_bias=not bn, name=name+'_deconv')
    if bn:
        h = PF.batch_normalization(h, batch_stat=not test, name=name+'_bn')
    if dropout and not test:
        h = F.dropout(h, 0.5)
    if relu is not None:
        if relu is F.relu:
            h = relu(h, inplace=True)
        else:
            h = relu(h)
    return h


def encoder(inp, test=False):
    with nn.parameter_scope('encoder'):
        c1 = PF.convolution(inp, 64, (3, 3), pad=(1, 1), name='c1')
        c1 = F.elu(c1)
        c2 = conv_bn_relu(c1, 128, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c2')
        c3 = conv_bn_relu(c2, 256, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c3')
        c4 = conv_bn_relu(c3, 512, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c4')
        c5 = conv_bn_relu(c4, 512, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c5')
        c6 = conv_bn_relu(c5, 512, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c6')
        c7 = conv_bn_relu(c6, 512, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c7')
        c8 = conv_bn_relu(c7, 512, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c8')
    return [c8, c7, c6, c5, c4, c3, c2, c1]


def decoder(inps, out_ch=3, test=False):
    with nn.parameter_scope('decoder'):
        c1 = conv_bn_relu(inps[0], 512, (4, 4), stride=(2, 2), pad=(
            1, 1), deconv=True, dropout=True, relu=F.relu, test=test, name='c1')
        c2 = conv_bn_relu((c1, inps[1]), 512, (4, 4), stride=(2, 2), pad=(
            1, 1), deconv=True, dropout=True, relu=F.relu, test=test, name='c2')
        c3 = conv_bn_relu((c2, inps[2]), 512, (4, 4), stride=(2, 2), pad=(
            1, 1), deconv=True, dropout=True, relu=F.relu, test=test, name='c3')
        c4 = conv_bn_relu((c3, inps[3]), 512, (4, 4), stride=(2, 2), pad=(
            1, 1), deconv=True, relu=F.relu, test=test, name='c4')
        c5 = conv_bn_relu((c4, inps[4]), 256, (4, 4), stride=(2, 2), pad=(
            1, 1), deconv=True, relu=F.relu, test=test, name='c5')
        c6 = conv_bn_relu((c5, inps[5]), 128, (4, 4), stride=(2, 2), pad=(
            1, 1), deconv=True, relu=F.relu, test=test, name='c6')
        c7 = conv_bn_relu((c6, inps[6]), 64, (4, 4), stride=(2, 2), pad=(
            1, 1), deconv=True, relu=F.relu, test=test, name='c7')
        c8 = F.tanh(PF.convolution(c7, out_ch, (3, 3), pad=(1, 1), name='c8'))
    return c8


def generator(inp, test=False):
    with nn.parameter_scope('generator'):
        z = encoder(inp, test=test)
        y = decoder(z, test=test)
    return y


def discriminator(inp1, inp2, patch_gan=True, test=False):
    with nn.parameter_scope('discriminator'):
        c0_1 = conv_bn_relu(inp1, 32, (4, 4), stride=(
            2, 2), pad=(1, 1), bn=False, relu=F.elu, name='c0_1')
        c0_2 = conv_bn_relu(inp2, 32, (4, 4), stride=(
            2, 2), pad=(1, 1), bn=False, relu=F.elu, name='c0_2')
        c1 = conv_bn_relu((c0_1, c0_2), 128, (4, 4), stride=(
            2, 2), pad=(1, 1), relu=F.elu, test=test, name='c1')
        c2 = conv_bn_relu(c1, 256, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c2')
        c3 = conv_bn_relu(c2, 512, (4, 4), stride=(2, 2), pad=(
            1, 1), relu=F.elu, test=test, name='c3')
        c4 = PF.convolution(c3, 1, (3, 3), stride=(1, 1),
                            pad=(1, 1), name='c4')
        if not patch_gan:
            c4 = F.mean(c4, axis=(2, 3))  # global average pooling
    return c4
