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

import numpy as np
import math

import xception_65 as xception


def atrous_spatial_pyramid_pooling(x, output_stride, test=False, fix_params=False):
    if output_stride not in [8, 16]:
        raise ValueError('output_stride neither 8 nor 16.')

    depth = 256
    atrous_rates = [6, 12, 18]
    if output_stride == 8:
        atrous_rates = [2*rate for rate in atrous_rates]

    with nn.parameter_scope("aspp0"):
        atrous_conv0 = PF.convolution(
            x, depth, (1, 1), with_bias=False, fix_parameters=fix_params)
        atrous_conv0 = F.relu(PF.batch_normalization(
            atrous_conv0, batch_stat=not test, fix_parameters=fix_params))

    atrous_conv = []
    for i in range(3):
        with nn.parameter_scope("aspp"+str(i+1)):
            atrous_conv.append(xception.separable_conv_with_bn(x, depth, stride=False, aspp=True,
                                                               atrous_rate=atrous_rates[i], last_block=True, eps=1e-05, test=test, fix_params=fix_params))

    # Image-level features
    with nn.parameter_scope("image_pooling"):
        x_shape = x.shape[2]
        h = F.average_pooling(
            x, (x.shape[2], x.shape[2]), stride=(x.shape[2], x.shape[2]))

        h = PF.convolution(h, depth, (1, 1), with_bias=False,
                           fix_parameters=fix_params)
        h = F.relu(PF.batch_normalization(
            h, batch_stat=not test, fix_parameters=fix_params))

        h = F.interpolate(h, output_size=(x_shape, x_shape), mode='linear')

    with nn.parameter_scope("concat_projection"):
        h5 = F.concatenate(
            h, atrous_conv0, atrous_conv[0], atrous_conv[1], atrous_conv[2], axis=1)

    return h5


def decoder(x, upsampled, num_classes, test=False, fix_params=False):

    # Project low-level features
    with nn.parameter_scope("feature_projection0"):
        h = PF.convolution(x, 48, (1, 1), with_bias=False,
                           fix_parameters=fix_params)
        h = F.relu(PF.batch_normalization(
            h, batch_stat=not test, fix_parameters=fix_params))

    h = F.concatenate(upsampled, h, axis=1)

    for i in range(2):
        with nn.parameter_scope("decoder_conv"+str(i)):
            h = xception.separable_conv_with_bn(
                h, 256, last_block=True, eps=1e-05, out=True, test=test, fix_params=fix_params)

    with nn.parameter_scope("logits/affine"):
        h = PF.convolution(h, num_classes, (1, 1), with_bias=True,
                           fix_parameters=fix_params)  # no activation

    with nn.parameter_scope("upsample2"):
        h = F.interpolate(h, output_size=(
            h.shape[2]*4 - 3, h.shape[2]*4 - 3), mode='linear')

    return h


def deeplabv3plus_model(x, output_stride, num_classes, test=False, fix_params=False):
    '''Encoder 
    '''
    # Get decoder endpoints from backbone
    endpoints = xception.xception_65(x, test=test, fix_params=fix_params)
    low_level_features = endpoints['Decoder End Point 1']

    encoder_output = atrous_spatial_pyramid_pooling(
        endpoints['Decoder End Point 2'], output_stride, test=test, fix_params=fix_params)

    with nn.parameter_scope("concat_projection"):
        encoder_output = PF.convolution(
            encoder_output, 256, (1, 1), with_bias=False, fix_parameters=fix_params)
        encoder_output = F.relu(PF.batch_normalization(
            encoder_output, batch_stat=not test, fix_parameters=fix_params))

    '''Decoder 
    '''
    with nn.parameter_scope("decoder"):
        with nn.parameter_scope("upsample1"):
            upsampled = F.interpolate(encoder_output, output_size=(
                low_level_features.shape[2], low_level_features.shape[2]), mode='linear')

        h = decoder(low_level_features, upsampled, num_classes,
                    test=test, fix_params=fix_params)

    return h
