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

from __future__ import absolute_import
from six.moves import range

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def dil_conv_3x3(x, output_filter, scope,
                 input_node_id, is_reduced, test, is_search):
    """
        dilated depthwise separable convolution with kernel 3x3.
    """
    if is_reduced and input_node_id < 2:
        stride = (2, 2)
    else:
        stride = (1, 1)

    h = F.relu(x)
    with nn.parameter_scope(scope + "depthwise"):
        h = PF.convolution(h, h.shape[1], kernel=(3, 3), pad=(2, 2),
                           stride=stride, dilation=(2, 2),
                           group=h.shape[1], with_bias=False)
    with nn.parameter_scope(scope + "pointwise"):
        h = PF.convolution(h, output_filter, (1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test,
                                   fix_parameters=is_search)
    return h


def dil_conv_5x5(x, output_filter, scope,
                 input_node_id, is_reduced, test, is_search):
    """
        dilated depthwise separable convolution with kernel 3x3.
    """
    if is_reduced and input_node_id < 2:
        stride = (2, 2)
    else:
        stride = (1, 1)

    h = F.relu(x)
    with nn.parameter_scope(scope + "depthwise"):
        h = PF.convolution(h, h.shape[1], kernel=(5, 5), pad=(4, 4),
                           stride=stride, dilation=(2, 2),
                           group=h.shape[1], with_bias=False)
    with nn.parameter_scope(scope + "pointwise"):
        h = PF.convolution(h, output_filter, (1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test,
                                   fix_parameters=is_search)
    return h


def sep_conv_3x3(x, output_filter, scope,
                 input_node_id, is_reduced, test, is_search):
    """
        depthwise separable convolution with kernel 3x3.
    """
    if is_reduced and input_node_id < 2:
        stride = (2, 2)
    else:
        stride = (1, 1)
    input_filter = x.shape[1]

    with nn.parameter_scope(scope + "_0"):
        h = F.relu(x)
        with nn.parameter_scope("depthwise"):
            h = PF.convolution(h, input_filter, kernel=(3, 3),
                               pad=(1, 1), stride=stride,
                               group=input_filter, with_bias=False)
        with nn.parameter_scope("pointwise"):
            h = PF.convolution(h, input_filter, (1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test,
                                       fix_parameters=is_search)

    with nn.parameter_scope(scope + "_1"):
        h = F.relu(h)
        with nn.parameter_scope("depthwise"):
            h = PF.convolution(h, input_filter, kernel=(3, 3),
                               pad=(1, 1), stride=(1, 1),
                               group=input_filter, with_bias=False)
        with nn.parameter_scope("pointwise"):
            h = PF.convolution(h, output_filter, (1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test,
                                       fix_parameters=is_search)

    return h


def sep_conv_5x5(x, output_filter, scope,
                 input_node_id, is_reduced, test, is_search):
    """
        depthwise separable convolution with kernel 5x5.
    """
    if is_reduced and input_node_id < 2:
        stride = (2, 2)
    else:
        stride = (1, 1)
    input_filter = x.shape[1]

    with nn.parameter_scope(scope + "_0"):
        h = F.relu(x)
        with nn.parameter_scope("depthwise"):
            h = PF.convolution(h, input_filter, kernel=(5, 5),
                               pad=(2, 2), stride=stride,
                               group=input_filter, with_bias=False)
        with nn.parameter_scope("pointwise"):
            h = PF.convolution(h, input_filter, (1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test,
                                       fix_parameters=is_search)

    with nn.parameter_scope(scope + "_1"):
        h = F.relu(h)
        with nn.parameter_scope("depthwise"):
            h = PF.convolution(h, input_filter, kernel=(5, 5),
                               pad=(2, 2), stride=(1, 1),
                               group=input_filter, with_bias=False)
        with nn.parameter_scope("pointwise"):
            h = PF.convolution(h, output_filter, (1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test,
                                       fix_parameters=is_search)

    return h


def max_pool_3x3(x, output_filter, scope,
                 input_node_id, is_reduced, test, is_search):
    """
        max pooling (with no spatial downsampling).
    """
    if is_reduced and input_node_id < 2:
        stride = (2, 2)
    else:
        stride = (1, 1)

    h = F.max_pooling(x, kernel=(3, 3), stride=stride, pad=(1, 1))
    with nn.parameter_scope(scope + "bn"):
        h = PF.batch_normalization(h, batch_stat=not test,
                                   fix_parameters=is_search)

    return h


def avg_pool_3x3(x, output_filter, scope,
                 input_node_id, is_reduced, test, is_search):
    """
        average pooling (with no spatial downsampling).
    """
    if is_reduced and input_node_id < 2:
        stride = (2, 2)
    else:
        stride = (1, 1)

    h = F.average_pooling(x, kernel=(3, 3), stride=stride, pad=(1, 1))
    with nn.parameter_scope(scope + "bn"):
        h = PF.batch_normalization(h, batch_stat=not test,
                                   fix_parameters=is_search)
    return h


def identity(x, output_filter, scope,
             input_node_id, is_reduced, test, is_search):
    """
        identity operation, i.e. Input does not change.
    """
    if is_reduced and input_node_id < 2:
        h = factorized_reduction(x, output_filter, scope, test, is_search)
    else:
        h = x
    return h


def zero(x, output_filter, scope,
         input_node_id, is_reduced, test, is_search):
    """
        Zero operation, i.e. all elements become 0.
    """
    if is_reduced and input_node_id < 2:
        h = F.max_pooling(x, kernel=(1, 1), stride=(2, 2))  # downsampling
        h = F.mul_scalar(h, 0)
    else:
        h = F.mul_scalar(x, 0)
    return h


def factorized_reduction(x, output_filter, scope, test, is_search):
    """
        Applying spatial reduction to input variable.
    """
    assert output_filter % 2 == 0
    x = F.relu(x)
    with nn.parameter_scope(scope):
        with nn.parameter_scope("conv_1"):
            conv_1 = PF.convolution(x, output_filter // 2,
                                    (1, 1), pad=None, stride=(2, 2),
                                    with_bias=False)

        conv_2 = F.pad(x, (0, 1, 0, 1), mode='constant')
        conv_2 = F.slice(conv_2, (0, 0, 1, 1))

        with nn.parameter_scope("conv_2"):
            conv_2 = PF.convolution(conv_2, output_filter // 2,
                                    (1, 1), pad=None, stride=(2, 2),
                                    with_bias=False)

        final_conv = F.concatenate(conv_1, conv_2, axis=1)

        with nn.parameter_scope("reduction_bn"):
            final_conv = PF.batch_normalization(
                    final_conv, batch_stat=not test, fix_parameters=is_search)
    return final_conv
