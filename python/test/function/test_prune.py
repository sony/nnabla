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

ctxs = list_context('Prune')


def ref_func_prune(x, rate):

    n = x.size
    shape = x.shape
    x = x.flatten()
    thresh_idx = int((n - 1) * rate)
    thresh_val = np.sort(np.abs(x))[thresh_idx]
    thresh_val = thresh_val + 1.0 if rate == 1.0 else thresh_val
    idx = np.where(np.abs(x) < thresh_val)
    y = np.copy(x)
    y[idx] = 0.0
    y = y.reshape(shape)
    return y


def ref_grad_prune(x, dy, rate, **kw):
    # pass through gradient from output to input
    return dy.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("rate", [0.0, 0.85, 0.9, 0.95, 1.0])
def test_prune_forward_backward(rate, seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    function_tester(rng, F.prune, ref_func_prune, inputs, func_args=[rate],
                    ctx=ctx, func_name=func_name,
                    ref_grad=ref_grad_prune,
                    disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("rate", [0.0, 0.85, 0.9, 0.95, 1.0])
def test_prune_double_backward(rate, seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]

    backward_function_tester(rng, F.prune, inputs, func_args=[rate], ctx=ctx)
