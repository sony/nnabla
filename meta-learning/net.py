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

""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I


def net(input, label, bn_batch_stat, args, init_params=None):
    output = forward_conv(input, bn_batch_stat, args, init_params)
    loss = loss_func(output, label)
    output2 = output.get_unlinked_variable(need_grad=False)
    accuracy = 1.0 - F.mean(F.top_n_error(output2, label, n=1))

    return (loss, accuracy)

# Network construction functions


def forward_conv(inp, bn_batch_stat, args, init_params, activation=F.relu):
    hidden1 = conv_block(inp, 'layer1', bn_batch_stat,
                         activation, args, init_params)
    hidden2 = conv_block(hidden1, 'layer2', bn_batch_stat,
                         activation, args, init_params)
    hidden3 = conv_block(hidden2, 'layer3', bn_batch_stat,
                         activation, args, init_params)
    hidden4 = conv_block(hidden3, 'layer4', bn_batch_stat,
                         activation, args, init_params)

    if args.datasource != 'omniglot' or args.method != 'maml':
        # hidden4 = F.reshape(hidden4, (hidden4.d.shape[0], -1), inplace=False)
        pass
    else:
        hidden4 = F.mean(hidden4, (2, 3))

    if init_params is None or 'layer5/affine/W' not in init_params:
        output = PF.affine(hidden4, args.num_classes, name='layer5')
    else:
        output = F.affine(
            hidden4, init_params['layer5/affine/W'], init_params['layer5/affine/b'])
    return output


def conv_block(inp, layer_name, bn_batch_stat, activation, args, init_params):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    k = 3
    stride, no_stride = (2, 2), (1, 1)
    pad = (1, 1)

    if init_params is None or layer_name + '/conv/W' not in init_params:
        if args.max_pool:
            conv_output = PF.convolution(
                inp, args.num_filters, (k, k), pad=pad, stride=no_stride, name=layer_name)
        else:
            conv_output = PF.convolution(
                inp, args.num_filters, (k, k), pad=pad, stride=stride, name=layer_name)
        normed = normalize(conv_output, layer_name,
                           bn_batch_stat, activation, args, init_params)
    else:
        if args.max_pool:
            conv_output = F.convolution(
                inp, init_params[layer_name + '/conv/W'], init_params[layer_name + '/conv/b'], pad=pad, stride=no_stride)
        else:
            conv_output = F.convolution(
                inp, init_params[layer_name + '/conv/W'], init_params[layer_name + '/conv/b'], pad=pad, stride=stride)
        normed = normalize(conv_output, layer_name,
                           bn_batch_stat, activation, args, init_params)

    if args.max_pool:
        normed = F.max_pooling(normed, stride, stride=stride)
    return normed


def normalize(inp, layer_name, bn_batch_stat, activation, args, init_params):
    if args.norm == 'batch_norm':
        if init_params is None:
            inp = PF.batch_normalization(
                inp, batch_stat=bn_batch_stat, name=layer_name)
        else:
            inp = F.batch_normalization(inp, init_params[layer_name + '/bn/beta'], init_params[layer_name + '/bn/gamma'],
                                        mean=None, variance=None, batch_stat=bn_batch_stat)

    if activation is not None:
        return activation(inp)
    else:
        return inp


def loss_func(pred, label):
    return F.mean(F.softmax_cross_entropy(pred, label))
