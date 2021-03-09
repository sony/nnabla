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

ctxs = list_context('Sign')


def ref_func_sign(x, alpha):
    # returns -1 if x < 0, alpha if x==0, 1 if x > 0.
    y = np.sign(x)
    y[x == 0] = alpha
    return y


def ref_grad_sign(x, dy, alpha, **kw):
    # pass through gradient from output to input
    return dy.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("alpha", [0.0, -1.0, 1.0])
def test_sign_forward_backward(seed, alpha, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    function_tester(rng, F.sign, ref_func_sign, inputs, func_args=[alpha],
                    ctx=ctx, func_name=func_name,
                    atol_b=1e-2, dstep=1e-3, ref_grad=ref_grad_sign)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("alpha", [0.0, -1.0, 1.0])
def test_sign_double_backward(seed, alpha, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    backward_function_tester(rng, F.sign, inputs, func_args=[alpha],
                             ctx=ctx,
                             dstep=1e-3)
