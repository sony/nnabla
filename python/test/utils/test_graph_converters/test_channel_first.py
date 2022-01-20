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

from nnabla.ext_utils import get_extension_context

from .ref_graphs.resnets import small_cl_resnet, small_cf_resnet
from nbla_test_utils import list_context

ctxs = list_context('Convolution')  # proxy to switch the context
batch_size = 1
resnet_ref = small_cf_resnet


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize('seed', [313])
@pytest.mark.parametrize('test', [True, False])
@pytest.mark.parametrize('graph_ref, graph_act', [(resnet_ref, small_cl_resnet)])
def test_channel_first(ctx, func_name, seed, test, graph_ref, graph_act):
    from .graph_converter_test_utils import structure_tester, value_tester

    if not func_name.endswith('Cudnn'):
        pytest.skip(
            "ChannelFirst/Last conversion is only supported in cuDNN context.")

    with nn.context_scope(ctx):
        # Random number
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        # Graph
        x_data = rng.randn(batch_size, 32, 32, 3)
        x = nn.Variable.from_numpy_array(x_data)

        y_tgt = graph_act(x, test=test)

        # FunctionModifier
        modifiers = []
        inputs = [x]
        outputs = [y_tgt]
        modifiers.append(GC.ChannelFirstModifier(inputs))

        y_act = GC.GraphConverter(modifiers).convert(outputs)

        # Ref Graph
        x_data_cf = rng.randn(batch_size, 3, 32, 32)
        x_cf = nn.Variable.from_numpy_array(x_data_cf)

        y_ref = graph_ref(x_cf, test=test)

        # Test
        structure_tester(y_ref, y_act[0])
        value_tester(y_tgt, y_act[0], rtol=6e-02, atol=5e-02)
