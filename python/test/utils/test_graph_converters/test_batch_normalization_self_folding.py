# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from .ref_graphs.resnets import small_cf_resnet, small_bn_self_folding_resnet


batch_size = 1
resnet_ref = small_bn_self_folding_resnet


@pytest.mark.parametrize('seed', [313])
@pytest.mark.parametrize('test', [True])
@pytest.mark.parametrize('graph_ref, graph_act', [(resnet_ref, small_cf_resnet)])
def test_batch_normalization_self_folding(seed, test, graph_ref, graph_act):
    from .graph_converter_test_utils import structure_tester, value_tester

    # Random number
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Graph
    x_data = rng.randn(batch_size, 3, 32, 32)
    x = nn.Variable.from_numpy_array(x_data)

    y_tgt = graph_act(x, test=test)

    # FunctionModifier
    modifiers = []
    modifiers.append(GC.BatchNormalizationSelfFoldingModifier())

    y_act = GC.GraphConverter(modifiers).convert(y_tgt)

    # Ref Graph
    y_ref = graph_ref(x, name='bn-self-folding-graph-ref')

    # Test
    structure_tester(y_ref, y_act)
    value_tester(y_tgt, y_act, rtol=6e-02, atol=5e-02)
