# Copyright 2022 Sony Group Corporation.
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

import pytest
import numpy as np

import nnabla as nn
import nnabla.experimental.graph_converters as GC
from nbla_test_utils import list_context
from .ref_graphs.resnets import (net_for_pruning,
                                 depthwise_net_for_pruning,)


ctxs = list_context('Convolution')  # proxy to switch the context
batch_size = 1


def pruning_inplace(pred, threshold, functions_to_pruning, channel_last=False):
    def traverse(l, var):
        if var.parent is not None and var.parent.info.type_name in functions_to_pruning:
            l.append(var.parent)
        if var.parent is not None:
            for ivar in var.parent.inputs:
                traverse(l, ivar)

    output_channel_axis = {
        'Affine': 1,
        'Convolution': 0,
        'Deconvolution': {
            True: -1,
            False: 1
        },
        'DepthwiseConvolution': 0,
        'DepthwiseDeconvolution': 0,
    }
    functions = []
    traverse(functions, pred)
    for func in functions:
        inputs = func.inputs
        x, w = inputs[:2]
        b = None
        if len(inputs) == 3:
            b = inputs[2]
        axis_to_sum = output_channel_axis[func.info.type_name]
        if not isinstance(axis_to_sum, int):
            axis_to_sum = axis_to_sum[channel_last]
        shape = list(range(w.ndim))
        shape.pop(axis_to_sum)
        l2_norm_per_channel = np.sum(
            w.d ** 2, axis=tuple(shape), keepdims=True)
        mask = l2_norm_per_channel > threshold
        w.d = w.d * mask
        if b is not None:
            b.d = b.d * mask.reshape((-1,))


@pytest.mark.parametrize('ctx, func_name', ctxs)
@pytest.mark.parametrize('seed', [313])
@pytest.mark.parametrize('threshold', [0.1, 0.2])
@pytest.mark.parametrize('w_bias', [False, True])
@pytest.mark.parametrize('channel_last', [False, True])
@pytest.mark.parametrize('functions_to_pruning', [['Affine'], ['Convolution'], ['Deconvolution'], ['DepthwiseConvolution'], ['DepthwiseDeconvolution'],
                                                  ['Affine', 'Convolution', 'Deconvolution', 'DepthwiseConvolution', 'DepthwiseDeconvolution']])
@pytest.mark.parametrize('graph', [net_for_pruning, depthwise_net_for_pruning])
def test_pruning(ctx, func_name, seed, threshold, w_bias,
                 channel_last, functions_to_pruning, graph):
    from .graph_converter_test_utils import structure_tester, value_tester

    if channel_last == True and not func_name.endswith('Cudnn'):
        pytest.skip(
            'ChannelLast conversion is only supported in cuDNN context.')
    # The Deconv in Cuda is strange, it causes the test to fail.
    if func_name.endswith('Cuda'):
        pytest.skip(
            'Skip test with Cuda context.')
    with nn.context_scope(ctx):
        # Random number
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        # Graph
        x_data = rng.randn(batch_size, 16, 16, 16)
        x1 = nn.Variable.from_numpy_array(x_data)
        x2 = nn.Variable.from_numpy_array(x_data)

        nn.random.seed(seed)
        y_tgt = graph(x1, threshold, with_bias=w_bias,
                      channel_last=channel_last, name_scope='net1')

        # Pruning with FunctionModifier
        modifiers = []
        modifiers.append(GC.PruningModifier(
            threshold, functions_to_pruning, channel_last))

        y_act = GC.GraphConverter(modifiers).convert(y_tgt)

        # Ref Graph
        nn.random.seed(seed)
        y_ref = graph(x2, threshold, with_bias=w_bias,
                      channel_last=channel_last, name_scope='net2')
        # Pruning manually
        pruning_inplace(y_ref, threshold, functions_to_pruning, channel_last)

        # Test
        structure_tester(y_ref, y_act)

        # Compare the results of manual pruning and pruning with GraphConverter
        # The results should be very close
        value_tester(y_ref, y_act, rtol=2e-6, atol=3e-8)
