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
import nnabla.parametric_functions as PF
import nnabla.functions as F
from nnabla.logger import logger
import numpy as np

endpoints = {}


def block(x, sh, num_units, stride, end_point, depth_list, sh_cut, shortcut_channels, act_fn=True, atrous_conv=False, atrous_rate=1, res=False, last_block=False, test=False, fix_params=False):

    for i in range(0, num_units):

        with nn.parameter_scope("unit_"+str(i+1)):

            x = unit(x, depth_list, stride, end_point, act_fn=act_fn, atrous_conv=atrous_conv,
                     atrous_rate=atrous_rate, last_block=last_block, test=test, fix_params=fix_params)

            if sh_cut == True:
                sh = shortcut(sh, shortcut_channels, not atrous_conv,
                              test=test, fix_params=fix_params)
                x = x+sh
            if res == True:
                x = x+sh
                sh = x

    return x


def unit(x, depth_list, stride, end_point, act_fn=True, atrous_conv=False, atrous_rate=1, last_block=False, test=False, fix_params=False):

    if last_block == False:
        x = F.relu(x)
    for i in range(0, len(depth_list)):
        if i == 2:
            act_fn = False
        with nn.parameter_scope("separable_conv"+str(i+1)):

            if end_point == True and i == 1:
                x = separable_conv_with_bn(x, depth_list[i], stride=False, aspp=False, atrous_rate=atrous_rate,
                                           act_fn=act_fn, last_block=last_block, end_point=end_point, test=test, fix_params=fix_params)

            else:
                if stride == True and i == 2:
                    x = separable_conv_with_bn(x, depth_list[i], stride=True, aspp=atrous_conv, atrous_rate=atrous_rate,
                                               act_fn=act_fn, last_block=last_block, test=test, fix_params=fix_params)
                else:
                    x = separable_conv_with_bn(x, depth_list[i], stride=False, aspp=atrous_conv, atrous_rate=atrous_rate,
                                               act_fn=act_fn, last_block=last_block, test=test, fix_params=fix_params)

    return x


def shortcut(x, f, stride=True, test=False, fix_params=False):

    with nn.parameter_scope("shortcut"):
        if(stride == False):
            h = PF.convolution(x, f, (1, 1), with_bias=False,
                               fix_parameters=fix_params)
        else:
            h = PF.convolution(x, f, (1, 1), stride=(
                2, 2), with_bias=False, fix_parameters=fix_params)
        h = PF.batch_normalization(
            h, batch_stat=not test, eps=1e-03, fix_parameters=fix_params)
    return h


def separable_conv_with_bn(x, f, stride=False, aspp=False, atrous_rate=1, act_fn=True, last_block=False, end_point=False, eps=1e-03, out=False, test=False, fix_params=False):

    with nn.parameter_scope("depthwise"):
        if(stride == True):
            h = PF.depthwise_convolution(x, (3, 3), stride=(2, 2), pad=(
                1, 1), with_bias=False, fix_parameters=fix_params)
        elif(aspp == True):
            h = PF.depthwise_convolution(x, (3, 3), pad=(atrous_rate, atrous_rate), stride=(
                1, 1), dilation=(atrous_rate, atrous_rate), with_bias=False, fix_parameters=fix_params)

        else:
            h = PF.depthwise_convolution(x, (3, 3), pad=(
                1, 1), with_bias=False, fix_parameters=fix_params)

        h = PF.batch_normalization(
            h, batch_stat=not test, eps=eps, fix_parameters=fix_params)
        if last_block == True:
            h = F.relu(h)

    with nn.parameter_scope("pointwise"):
        h = PF.convolution(h, f, (1, 1), stride=(
            1, 1), with_bias=False, fix_parameters=fix_params)
        h = PF.batch_normalization(
            h, batch_stat=not test, eps=eps, fix_parameters=fix_params)
        if end_point == True:
            global endpoints
            endpoints['Decoder End Point 1'] = h

        if act_fn == True:
            h = F.relu(h)

    return h


def entry_flow(x, num_blocks, depth_list, test=False, fix_params=False):

    shortcut_channels = [128, 256, 728]
    global endpoints

    with nn.parameter_scope("1"):
        h = PF.convolution(x, 32, (3, 3), pad=(1, 1), stride=(
            2, 2), with_bias=False, fix_parameters=fix_params)
        h = F.relu(PF.batch_normalization(h, batch_stat=not test,
                                          eps=1e-03, fix_parameters=fix_params))

    with nn.parameter_scope("2"):
        h = PF.convolution(h, 64, (3, 3), pad=(
            1, 1), with_bias=False, fix_parameters=fix_params)
        h = F.relu(PF.batch_normalization(h, batch_stat=not test,
                                          eps=1e-03, fix_parameters=fix_params))

    x = h
    sh = x
    for i in range(0, num_blocks):
        with nn.parameter_scope("block"+str(i+1)):

            if i == 1:
                x = block(x, sh, 1, True, True,
                          depth_list[i], True, shortcut_channels[i], test=test)
            else:
                x = block(x, sh, 1, True, False,
                          depth_list[i], True, shortcut_channels[i], test=test)
            sh = x
        if i == 2:
            endpoints['conv1'] = x

    return x


def middle_flow(x, num_blocks, depth_list, test=False, fix_params=False):

    shortcut_channels = [0]
    sh = x
    for i in range(0, num_blocks):
        with nn.parameter_scope("block"+str(i+1)):
            x = block(x, sh, 16, False, False,
                      depth_list[i], False, shortcut_channels, res=True, test=test)
            sh = x

    return x


def exit_flow(x, num_blocks, depth_list, test=False, fix_params=False):

    shortcut_channels = [1024, 0]
    sh = x
    for i in range(0, num_blocks):
        with nn.parameter_scope("block"+str(i+1)):
            if i == 0:
                x = block(x, sh, 1, False, False,
                          depth_list[i], True, shortcut_channels[i], atrous_conv=True, test=test)

            else:
                x = block(x, sh, 1, False, False, depth_list[i], False, shortcut_channels[i],
                          atrous_conv=True, atrous_rate=2, last_block=True, test=test)

            sh = x

    return x


def xception_65(x, test=False, fix_params=False):

    entry_flow_depth_list = [[128, 128, 128], [256, 256, 256], [728, 728, 728]]
    middle_flow_depth_list = [[728, 728, 728]]
    exit_flow_depth_list = [[728, 1024, 1024], [1536, 1536, 2048]]

    with nn.parameter_scope("xception_65"):
        with nn.parameter_scope("entry_flow"):
            x = entry_flow(x, 3, entry_flow_depth_list,
                           test=test, fix_params=fix_params)

        with nn.parameter_scope("middle_flow"):
            x = middle_flow(x, 1, middle_flow_depth_list,
                            test=test, fix_params=fix_params)

        with nn.parameter_scope("exit_flow"):
            x = exit_flow(x, 2, exit_flow_depth_list,
                          test=test, fix_params=fix_params)
            x = F.relu(x)
        global endpoints
        endpoints['Decoder End Point 2'] = x

    return endpoints
