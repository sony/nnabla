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

from .ref_graphs.resnets import (small_multiple_inputs_outputs_resnet,
                                 small_multiple_inputs_outputs_bn_folding_resnet)

batch_size = 1


@pytest.mark.parametrize('seed', [313])
@pytest.mark.parametrize('w_bias', [True, False])
@pytest.mark.parametrize('test', [True])
@pytest.mark.parametrize('graph_ref, graph_act', [(small_multiple_inputs_outputs_bn_folding_resnet,
                                                   small_multiple_inputs_outputs_resnet)])
def test_multi_inputs_outputs(seed, w_bias, test, graph_ref, graph_act):
    from .graph_converter_test_utils import structure_tester, value_tester

    # Random number
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Graph
    x_data = rng.randn(batch_size, 3, 32, 32)
    x = nn.Variable.from_numpy_array(x_data)

    z_data = rng.randn(batch_size, 3, 32, 32)
    z = nn.Variable.from_numpy_array(z_data)

    y_tgt = graph_act([x, z], w_bias=w_bias, test=test)

    # FunctionModifier
    modifiers = []
    modifiers.append(GC.BatchNormalizationFoldingModifier())

    y_act = GC.GraphConverter(modifiers).convert(y_tgt)

    # Ref Graph
    y_ref = graph_ref([x, z], test=test)

    # Test
    for ref, act, tgt in zip(y_ref, y_act, y_tgt):
        structure_tester(ref, act)
        value_tester(tgt, act, rtol=6e-02, atol=5e-02)
