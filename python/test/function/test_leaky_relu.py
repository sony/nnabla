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
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('LeakyReLU')


def ref_leaky_relu(x, alpha):
    return alpha * np.minimum(x, 0) + np.maximum(x, 0)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("alpha", [0, 0.1, -0.1])
def test_leaky_relu_forward_backward(seed, alpha, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3))]
    function_tester(rng, F.leaky_relu, ref_leaky_relu, inputs,
                    func_args=[alpha], ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("alpha", [0, 0.2])
def test_leaky_relu_inplace(seed, alpha, ctx, func_name):
    from nbla_test_utils import inplace_function_test_helper
    x = nn.Variable([2, 3, 4], need_grad=True)
    inplace_function_test_helper(
        [x], F.leaky_relu, ctx=ctx, rng=np.random.RandomState(seed))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("alpha", [0, 0.2])
def test_relu_double_backward(seed, alpha, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3))]
    backward_function_tester(rng, F.leaky_relu, None,
                             inputs=inputs,
                             func_args=[alpha], func_kwargs={},
                             atol_b=1e-3,
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx, func_name=None,
                             disable_half_test=True)

# TODO: inplace double_backward
