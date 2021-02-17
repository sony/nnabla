# Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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


def ref_top_k_data_fw(x, k, abs, reduce, base_axis):
    return ref_top_k_data(x, k, abs, reduce, base_axis)[0]


def ref_top_k_data_bw(x, g, k, abs, reduce, base_axis, **kw):
    return ref_top_k_data(x, k, abs, reduce, base_axis, g)[1].flatten()


@pytest.mark.parametrize("ctx, fname", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("abs", [False, True])
@pytest.mark.parametrize("reduce", [True, False])
@pytest.mark.parametrize("ishape, k, base_axis", [
    ((4, 5, 6), 1, 0), ((4, 5, 6), 1, 1), ((4, 5, 6), 1, 2), ((4, 5, 6), 1, -2),
    ((4, 5, 6), 5, 0), ((4, 5, 6), 5, 1), ((4, 5, 6), 5, 2), ((4, 5, 6), 5, -1),
    ((1, 1000), 10, 1), ((1, 100000), 1024, 1), ((
        1, 100000), 1025, 1), ((1, 100000), 1025, -2)
])
def test_forward_backward(seed, ishape, k, abs, reduce, base_axis, ctx, fname):
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*ishape).astype(np.float32)]
    function_tester(rng, F.top_k_data, ref_top_k_data_fw, inputs, ctx=ctx,
                    func_name=fname, ref_grad=ref_top_k_data_bw,
                    func_args=[k, abs, reduce, base_axis],
                    disable_half_test=k > 10)

    # Note: FP16 has too many duplicate value for larger K to get the
    # same sort order as FP32 and this makes the function tester fail
    # when comparing FP16 to FP32 results of gradient computation.
