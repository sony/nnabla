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

from .ref_graphs.resnets import small_bn_resnet
from .ref_graphs.lenets import lenet


@pytest.mark.parametrize('seed', [313])
@pytest.mark.parametrize('graph', [lenet, small_bn_resnet])
def test_no_grad(seed, graph):
    from .graph_converter_test_utils import structure_tester, value_tester

    nn.clear_parameters()

    # Random number
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Graph
    x_data = rng.randn(4, 3, 32, 32)
    x0 = nn.Variable.from_numpy_array(x_data)\
        .apply(need_grad=False) \
        .apply(persistent=True)
    y0 = graph(x0)
    y1 = y0.no_grad()

    # Test
    def assert_need_grad_flase(f):
        for inp in f.inputs:
            assert inp.need_grad == False, "need_grad must be false"
        for out in f.outputs:
            assert out.need_grad == False, "need_grad must be false"
    y1.visit(assert_need_grad_flase)
    structure_tester(y0, y1)
    value_tester(y0, y1, clear_no_need_grad=True)
