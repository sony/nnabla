# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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

ctxs = list_context('WarpByFlow')


def warp_by_flow(data, flow):
    N, C, H, W = data.shape
    out = np.ndarray(data.shape)
    for n in range(N):
        xx, yy = np.meshgrid(range(W), range(H))
        xf, yf = xx + flow[n, 0], yy + flow[n, 1]
        xL = np.maximum(0, np.minimum(W - 1, np.floor(xf))).astype(np.int)
        yT = np.maximum(0, np.minimum(H - 1, np.floor(yf))).astype(np.int)
        xR = np.maximum(0, np.minimum(W - 1, np.floor(xf + 1))).astype(np.int)
        yB = np.maximum(0, np.minimum(H - 1, np.floor(yf + 1))).astype(np.int)
        alpha = xf - np.floor(xf)
        beta = yf - np.floor(yf)
        for c in range(C):
            upper_left = data[n, c, yT, xL]
            lower_left = data[n, c, yB, xL]
            upper_right = data[n, c, yT, xR]
            lower_right = data[n, c, yB, xR]
            interp_upper = alpha * (upper_right - upper_left) + upper_left
            interp_lower = alpha * (lower_right - lower_left) + lower_left
            out[n, c] = beta * (interp_lower - interp_upper) + interp_upper
    return out


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("shape", [
    (1, 1, 4, 5), (4, 3, 4, 5), (2, 6, 1, 10), (8, 8, 1, 1)
])
def test_warp_by_flow_forward_backward(shape, seed, ctx, func_name):
    N, C, H, W = shape
    rng = np.random.RandomState(seed)
    data = rng.randn(N, C, H, W).astype(np.float32)
    flow = rng.randn(N, 2, H, W).astype(np.float32)
    function_tester(rng, F.warp_by_flow, warp_by_flow, [data, flow], ctx=ctx,
                    func_name=func_name, func_args=[], atol_b=1e-2)
