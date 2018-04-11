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

ctxs = list_context('TopKData')


def ref_top_k_data(x, k, abs, reduce, base_axis, grad=None):
    ns = np.prod(x.shape[:base_axis], dtype=int)  # number of samples
    ss = np.prod(x.shape[base_axis:], dtype=int)  # sample size
    xd = x.reshape(ns, ss).copy()
    ix = np.fliplr(np.argsort(np.abs(xd) if abs else xd)[:, -k:])
    yd = np.zeros((ns, k if reduce else ss))
    for idx, row in enumerate(ix):
        yd[idx, slice(None) if reduce else row] = xd[idx, row]
    if grad is not None and reduce is True:
        gg = grad.reshape(yd.shape).copy()
        xg = np.zeros(xd.shape)
        for idx, row in enumerate(ix):
            xg[idx, row] = gg[idx]
        xg = xg.reshape(x.shape)
    else:
        xg = grad
    yd = yd.squeeze(axis=0) if base_axis == 0 else yd
    return (yd.reshape(x.shape[:base_axis] + (k,) if reduce else x.shape), xg)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("abs", [False, True])
@pytest.mark.parametrize("reduce", [False, True])
@pytest.mark.parametrize("base_axis", [1, 0, 2])
def test_forward_backward(seed, k, abs, reduce, base_axis, ctx, func_name):
    rng = np.random.RandomState(seed)
    ishape = (4, 5, 6)

    x = nn.Variable.from_numpy_array(rng.randn(*ishape), need_grad=True)
    y = F.top_k_data(x, k, abs, reduce, base_axis)
    y.forward()

    x.grad.zero()
    grad = rng.randn(*y.shape)
    y.backward(grad)

    ref_d, ref_g = ref_top_k_data(x.d, k, abs, reduce, base_axis, grad)
    assert np.allclose(ref_d, y.d, atol=1e-6)
    assert np.allclose(ref_g, x.g, atol=1e-6)
