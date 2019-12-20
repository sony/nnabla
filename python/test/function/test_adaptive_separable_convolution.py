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

ctxs = list_context('AdaptiveSeparableConvolution')


def ref_adaptive_separable_convolution(x, kv, kh):
    B, C, H, W = x.shape
    KV = kv.shape[1]
    KH = kh.shape[1]
    oH, oW = H - KV + 1, W - KH + 1
    y = np.zeros([B, C, oH, oW])
    for b in range(B):
        for h in range(oH):
            for w in range(oW):
                approx_kernel = np.outer(kv[b, :, h, w], kh[b, :, h, w])
                y[b, :, h, w] = np.sum(
                    approx_kernel * x[b, :, h:h+KV, w:w+KH], axis=(1, 2))
    return y


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("x_shape, kv_filter, kh_filter", [
   ((1, 3, 8, 8), 4, 4),
   ((2, 3, 16, 8), 4, 2),
   ((2, 3, 8, 16), 2, 4),
])
def test_adaptive_separable_convolution_forward_backward(seed, ctx, func_name, x_shape, kv_filter, kh_filter):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)

    b, c, h, w = x_shape
    kv_shape = [b, kv_filter, h - kv_filter + 1, w - kh_filter + 1]
    kh_shape = [b, kh_filter, h - kv_filter + 1, w - kh_filter + 1]

    x = rng.randn(*x_shape)
    kv = rng.randn(*kv_shape)
    kh = rng.randn(*kh_shape)
    inputs = [x, kv, kh]

    backward = [True, True, True]
    function_tester(rng, F.adaptive_separable_convolution,
                    ref_adaptive_separable_convolution, inputs, backward=backward,
                    atol_f=1e-6, atol_b=3e-2, atol_accum=3e-2, dstep=1e-3, ctx=ctx, func_name=func_name)
