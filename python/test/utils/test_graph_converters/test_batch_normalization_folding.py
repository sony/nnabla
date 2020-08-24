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

from __future__ import absolute_import

import pytest
import numpy as np

import nnabla as nn
import nnabla.experimental.graph_converters as GC

from nnabla.ext_utils import get_extension_context
from nbla_test_utils import list_context
from .ref_graphs.lenets import bn_lenet, bn_folding_lenet, bn_opp_lenet
from .ref_graphs.resnets import (small_bn_resnet,
                                 small_bn_folding_resnet,
                                 small_bn_opp_resnet)


ctxs = list_context('Convolution')  # proxy to switch the context
batch_size = 1
lenet_ref = bn_folding_lenet
resnet_ref = small_bn_folding_resnet


@pytest.mark.parametrize('ctx, func_name', ctxs)
@pytest.mark.parametrize('seed', [313])
@pytest.mark.parametrize('test', [True])
@pytest.mark.parametrize('w_bias', [True])
@pytest.mark.parametrize('channel_last', [True])
@pytest.mark.parametrize('graph_ref, graph_act, opposite',
                         [(resnet_ref, small_bn_resnet, False),
                          (resnet_ref, small_bn_opp_resnet, True)])
def test_batch_normalization_folding(ctx, func_name, seed, test, w_bias,
                                     channel_last, graph_ref, graph_act, opposite):
    from .graph_converter_test_utils import structure_tester, value_tester

    if channel_last == True and not func_name.endswith('Cudnn'):
        pytest.skip(
            'ChannelLast conversion is only supported in cuDNN context.')

    with nn.context_scope(ctx):
        # Random number
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        # Graph
        x_data = rng.randn(batch_size, 32, 32, 3) if channel_last == True else rng.randn(
            batch_size, 3, 32, 32)
        x = nn.Variable.from_numpy_array(x_data)

        y_tgt = graph_act(x, test=test, w_bias=w_bias,
                          channel_last=channel_last)

        # FunctionModifier
        modifiers = []
        modifiers.append(GC.BatchNormalizationFoldingModifier(
            opposite, channel_last))

        y_act = GC.GraphConverter(modifiers).convert(y_tgt)

        # Ref Graph
        y_ref = graph_ref(x, test=test, channel_last=channel_last)

        # Test
        structure_tester(y_ref, y_act)
        value_tester(y_tgt, y_act, rtol=6e-02, atol=5e-02)
