# Copyright 2018,2019,2020,2021 Sony Corporation.
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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('TopKData')


def ref_top_k_data(x, k, abs, reduce, base_axis, largest, with_index,
                   grad=None):
    ns = np.prod(x.shape[:base_axis], dtype=int)  # number of samples
    ss = np.prod(x.shape[base_axis:], dtype=int)  # sample size
    xd = x.reshape(ns, ss).copy()
    sign = -1.0 if largest else 1.0
    ix = np.argsort(sign * (np.abs(xd) if abs else xd))[:, :k]
    yd = np.zeros((ns, k if reduce else ss))
    indices = np.full(yd.shape, -1)
    for idx, row in enumerate(ix):
        yd[idx, slice(None) if reduce else row] = xd[idx, row]
        indices[idx, slice(None) if reduce else row] = row
    if grad is not None and reduce is True:
        gg = grad.reshape(yd.shape).copy()
        xg = np.zeros(xd.shape)
        for idx, row in enumerate(ix):
            xg[idx, row] = gg[idx]
        xg = xg.reshape(x.shape)
    else:
        xg = grad
    yd = yd.squeeze(axis=0) if base_axis == 0 else yd
    yd = yd.reshape(x.shape[:base_axis] + (k,) if reduce else x.shape)
    indices = indices.reshape(yd.shape)
    if with_index:
        return ((yd, indices), xg)
    else:
        return (yd, xg)


def ref_top_k_data_fw(x, k, abs, reduce, base_axis, largest, with_index):
    return ref_top_k_data(x, k, abs, reduce, base_axis, largest, with_index)[0]


def ref_top_k_data_bw(x, g, k, abs, reduce, base_axis, largest, with_index,
                      **kw):
    assert not with_index
    return ref_top_k_data(
        x, k, abs, reduce, base_axis, largest, with_index, g)[1].flatten()

# reference function takes two grads of outputs when with_index = True
def ref_top_k_data_bw_with_index(x, gx, gi, k, abs, reduce, base_axis,
                                 largest, with_index, **kw):
    assert with_index
    return ref_top_k_data(
        x, k, abs, reduce, base_axis, largest, with_index, gx)[1].flatten()

@pytest.mark.parametrize("ctx, fname", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("abs", [False, True])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("reduce, with_index", [
    (False, False), (True, False), (True, True)
])
@pytest.mark.parametrize("ishape, k, base_axis", [
    ((4, 5, 6), 1, 0), ((4, 5, 6), 1, 1), ((4, 5, 6), 1, 2), ((4, 5, 6), 1, -2),
    ((4, 5, 6), 5, 0), ((4, 5, 6), 5, 1), ((4, 5, 6), 5, 2), ((4, 5, 6), 5, -1),
    ((1, 1000), 10, 1), ((1, 100000), 1024, 1), ((
        1, 100000), 1025, 1), ((1, 100000), 1025, -2)
])
def test_forward_backward(seed, ishape, k, abs, reduce, base_axis, largest,
                          with_index, ctx, fname):
    # Use unique random numbers because sorting algorithm for top-k 
    # is not stable.
    rng = np.random.RandomState(seed)
    n = np.prod(ishape)
    x = np.arange(n)
    x[np.arange(n, step=2)] *= -1  # Negate even number
    rng.shuffle(x)
    x = x.reshape(ishape).astype(np.float32)

    ref_grad = ref_top_k_data_bw_with_index if with_index else ref_top_k_data_bw
    function_tester(rng, F.top_k_data, ref_top_k_data_fw, [x], ctx=ctx,
                    func_name=fname, ref_grad=ref_grad,
                    func_args=[k, abs, reduce, base_axis, largest, with_index],
                    disable_half_test=k > 10)

    # Note: FP16 has too many duplicate value for larger K to get the
    # same sort order as FP32 and this makes the function tester fail
    # when comparing FP16 to FP32 results of gradient computation.
