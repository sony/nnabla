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

import pytest
import numpy as np
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('SigmoidCrossEntropy')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_sigmoid_cross_entropy_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(
        np.float32), rng.rand(2, 3, 4).astype(np.float32)]
    inputs[1] = np.round(inputs[1])
    function_tester(rng, F.sigmoid_cross_entropy,
                    lambda x, y: -
                    (y * np.log(1 / (np.exp(-x) + 1)) + (1 - y)
                     * np.log(1 - 1 / (np.exp(-x) + 1))),
                    inputs,
                    atol_b=1e-2, ctx=ctx, func_name=func_name, backward=[True, False])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_sigmoid_cross_entropy_double_backward(seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(
        np.float32), rng.rand(2, 3, 4).astype(np.float32)]
    inputs[1] = np.round(inputs[1])
    backward_function_tester(rng, F.sigmoid_cross_entropy,
                             inputs,
                             atol_accum=1e-2, dstep=1e-3,
                             ctx=ctx, backward=[True, False])
