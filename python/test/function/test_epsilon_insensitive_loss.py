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

ctxs = list_context('EpsilonInsensitiveLoss')


def ref_epsilon_insensitive_loss_forward(x0, x1, epsilon):
    y = np.zeros_like(x0)
    abs_diff = np.abs(x0 - x1)
    idx = np.where(abs_diff > epsilon)
    y[idx] = abs_diff[idx] - epsilon
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("epsilon", [0.001, 1])
def test_epsilon_insensitive_loss_forward_backward(seed, ctx, func_name, epsilon):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2 for _ in range(2)]
    function_tester(rng, F.epsilon_insensitive_loss,
                    ref_epsilon_insensitive_loss_forward, inputs,
                    func_args=[epsilon],
                    atol_b=1e-2, ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("epsilon", [0.001, 1])
def test_epsilon_insensitive_loss_double_backward(seed, ctx, func_name, epsilon):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2 for _ in range(2)]
    backward_function_tester(rng, F.epsilon_insensitive_loss,
                             inputs,
                             func_args=[epsilon],
                             atol_accum=5e-3, ctx=ctx)
