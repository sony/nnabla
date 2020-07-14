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
from utils.utils import bicubic_four


def conv2d(input, output_channels, kernel, stride, name='', pad=(1, 1), bias=True):
    """
    Convolution layer:
    """
    return PF.convolution(input, output_channels, kernel=kernel, stride=stride,
                          pad=pad, with_bias=bias, name=name, channel_last=True)


def deconv2d(input, output_channels, kernel, stride, name='', pad=(1, 1), output_padding=(1, 1), bias=True):
    """
    Deconvolution layer
    """
    return PF.deconvolution(input, output_channels, kernel=kernel, stride=stride, output_padding=output_padding,
                            pad=pad, with_bias=bias, name=name, channel_last=True)


# Definition of the flow_estimation : Flow-Estimation network
def flow_estimator(fnet_input):
    """
    Flow Estimator network
        Returns the flow estmation for given batch of consecutive frames.
    """
    def down_block(input, output_channels=64, stride=1, scope='down_block'):
        with nn.parameter_scope(scope):
            net = conv2d(input, output_channels, (3, 3),
                         (stride, stride), name='conv_1')
            net = F.leaky_relu(net, 0.2)
            net = conv2d(net, output_channels, (3, 3),
                         (stride, stride), name='conv_2')
            net = F.leaky_relu(net, 0.2)
            net = F.max_pooling(net, (2, 2), channel_last=True)
        return net

    def up_block(input, output_channels=64, stride=1, scope='up_block'):
        with nn.parameter_scope(scope):
            net = conv2d(input, output_channels, (3, 3),
                         (stride, stride), name='conv_1')
            net = F.leaky_relu(net, 0.2)
            net = conv2d(net, output_channels, (3, 3),
                         (stride, stride), name='conv_2')
            net = F.leaky_relu(net, 0.2)
            net = F.interpolate(net, scale=(2, 2), channel_last=True)
        return net

    with nn.parameter_scope('autoencode_unit'):
        net = down_block(fnet_input, 32, scope='encoder_1')
        net = down_block(net, 64, scope='encoder_2')
        net = down_block(net, 128, scope='encoder_3')

        net = up_block(net, 256, scope='decoder_1')
        net = up_block(net, 128, scope='decoder_2')
        net = up_block(net, 64, scope='decoder_3')
        with nn.parameter_scope('output_stage'):
            net = conv2d(net, 32, (3, 3), (1, 1), name='conv1')
            net = F.leaky_relu(net, 0.2)
            net = conv2d(net, 2, (3, 3), (1, 1), name='conv2')
            # the 24.0 is the max Velocity, details can be found in TecoGAN paper
            net = F.tanh(net) * 24.0
    return net


def generator(gen_input, gen_output_channels, num_resblock=16):
    """
    Generator network
        Returns the HR frames for given batch of LR frames
    """
    # The residual blocks
    def residual_block(input, output_channels=64, stride=1, scope='res_block'):
        with nn.parameter_scope(scope):
            net = conv2d(input, output_channels, (3, 3),
                         (stride, stride), name='conv_1')
            net = F.relu(net)
            net = conv2d(net, output_channels, (3, 3),
                         (stride, stride), name='conv_2')
            net = net + input
        return net

    with nn.parameter_scope('generator_unit'):
        # The input layer
        with nn.parameter_scope('input_stage'):
            net = conv2d(gen_input, 64, (3, 3), (1, 1), name='conv')
            net = F.relu(net)

        # The residual block parts
        for i in range(0, num_resblock, 1):  # should be 16 for TecoGAN, and 10 for TecoGANmini
            net = residual_block(net, 64, 1, scope='resblock_%d' % (i+1))

        with nn.parameter_scope('conv_tran2highres'):
            net = deconv2d(net, 64, (3, 3), (2, 2), name='conv_tran1')
            net = F.relu(net)
            net = deconv2d(net, 64, (3, 3), (2, 2), name='conv_tran2')
            net = F.relu(net)

        with nn.parameter_scope('output_stage'):
            net = conv2d(net, gen_output_channels, (3, 3), (1, 1), name='conv')
            low_res_in = gen_input[:, :, :, 0:3]  # ignore warped pre high res
            bicubic_hi = bicubic_four(low_res_in)
            net = net + bicubic_hi
            net = net * 2 - 1
    return net


def discriminator_block(input, out_channels, scope, kernel=(4, 4), stride=(2, 2)):
    with nn.parameter_scope(scope):
        net = conv2d(input, out_channels, kernel, stride, bias=False)
        net = PF.batch_normalization(net, eps=0.001, axes=[3], no_scale=True)
        net = F.leaky_relu(net, alpha=0.2, inplace=True)
    return net


def discriminator(dis_input):
    """
    Discriminator network
        Returns output from dicsriminator block 1 to 7
    """
    layer_list = []
    net = conv2d(dis_input, 64, (3, 3), (1, 1), name='conv')
    net = F.leaky_relu(net, alpha=0.2, inplace=True)        # (b,h,w,64)
    net = discriminator_block(net, 64, 'disblock_1')        # (b,h/2,w/2,64)
    layer_list += [net]
    net = discriminator_block(net, 64, 'disblock_3')        # (b,h/4,w/4,64)
    layer_list += [net]
    net = discriminator_block(net, 128, 'disblock_5')       # (b,h/8,w/8,64)
    layer_list += [net]
    net = discriminator_block(net, 256, 'disblock_7')       # (b,h/16,w/16,64)
    layer_list += [net]
    # channel wise affine
    net = PF.affine(net, 1, base_axis=3, name='affine0')
    net = F.sigmoid(net)                                    # (b,h/16,w/16,1)
    return net, layer_list
