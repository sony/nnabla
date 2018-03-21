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

ctxs = list_context('TopKSelect')


def ref_top_k_select(x, k, axis):
    xx = x.reshape(int(np.prod(x.shape[:axis])), int(np.prod(x.shape[axis:])))
    xx = np.abs(xx) if k < 0 else xx
    yy = np.zeros(xx.shape)
    ix = np.argsort(xx)[:, -np.abs(k):]
    for row, col in enumerate(ix):
        yy[row, col] = xx[row, col]
    return yy.reshape(x.shape)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("k", [0, 1, 5, -3])
def test_top_k_select_forward(seed, k, axis, ctx, func_name):
    rng = np.random.RandomState(seed)
    ishape = (4, 5, 6)

    x = nn.Variable.from_numpy_array(rng.rand(*ishape))
    y = F.top_k_select(x, k, 0, axis)
    y.forward()

    assert np.allclose(ref_top_k_select(x.d, k, axis), y.d, atol=1e-6)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("k", [0, 1, 5, -3])
def test_top_k_select_backward(seed, k, axis, ctx, func_name):
    rng = np.random.RandomState(seed)
    ishape = (4, 5, 6)

    x = nn.Variable(ishape, need_grad=True)
    y = F.top_k_select(x, 0, k, axis)
    grad = rng.rand(*ishape)
    x.grad.zero()
    y.backward(grad)

    assert np.allclose(ref_top_k_select(grad, k, axis), x.g, atol=1e-6)
