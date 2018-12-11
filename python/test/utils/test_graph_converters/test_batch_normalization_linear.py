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

import os
import pytest
import numpy as np

import nnabla as nn
import nnabla.experimental.graph_converters as GC
import nnabla.functions as F
import nnabla.parametric_functions as PF

from collections import OrderedDict

from ref_graphs.lenets import lenet, bn_linear_lenet
from ref_graphs.resnets import small_resnet, bn_linear_small_resnet


lenet_ref = bn_linear_lenet
small_resnet_ref = bn_linear_small_resnet


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("test", [True])
@pytest.mark.parametrize("graph_ref, graph_act",
                         [(lenet_ref, lenet),
                          (small_resnet_ref, small_resnet)
                          ])
def test_batch_normalization_linear(seed, test, graph_ref, graph_act):
    from graph_converter_test_utils import structure_tester, value_tester, print_params
    # because linearized parameters (c0, c1) are to be shared among models
    nn.clear_parameters()

    # Random number
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Graph
    x_data = rng.randn(4, 3, 32, 32)
    x = nn.Variable.from_numpy_array(x_data)
    y_tgt = graph_act(x, test=test)

    # Convert
    name = "bn-linear-graph"
    converter = GC.BatchNormalizationLinearConverter(name=name)
    y_act = converter.convert(y_tgt, [x])

    # Ref Graph
    name = "bn-linear-graph-ref"
    y_ref = graph_ref(x, name=name)

    # Test
    structure_tester(y_ref, y_act)
    value_tester(y_tgt, y_act)
