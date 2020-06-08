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

ctxs = list_context('FixedPointQuantize')


def ref_fixed_point_quantize(x, sign, n, delta, quantize, ste_fine_grained):
    assert (n > 0 and delta > 0)

    ref_delta = delta
    if sign:
        ref_max = ((1 << (n - 1)) - 1.0) * ref_delta
        ref_min = -ref_max
    else:
        ref_max = ((1 << n) - 1.0) * ref_delta
        ref_min = 0.

    if quantize:
        x_q = ((np.abs(x) / ref_delta + 0.5).astype(np.int32)
               * ref_delta) * np.sign(x)
        x_q[np.where(x > ref_max)] = ref_max
        x_q[np.where(x < ref_min)] = ref_min
    else:
        x_q = x
    return x_q


def ref_grad_fixed_point_quantize(x, dy, sign, n, delta,
                                  quantize, ste_fine_grained, **kw):

    if not ste_fine_grained:
        return dy.flatten()

    dx = dy.copy()
    ref_delta = delta
    if sign:
        ref_max = ((1 << (n - 1)) - 1.0) * ref_delta
        ref_min = -ref_max
    else:
        ref_max = ((1 << n) - 1.0) * ref_delta
        ref_min = 0.

    dy[np.where(x > ref_max)] = 0.
    dy[np.where(x < ref_min)] = 0.
    return dy.flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("sign", [False, True])
@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("delta", [2**-7, 0.005])
@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("ste_fine_grained", [False, True])
def test_fixed_point_quantize_forward_backward(seed,
                                               sign, n, delta, quantize,
                                               ste_fine_grained,
                                               ctx, func_name):
    from nbla_test_utils import cap_ignore_region, \
        function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(2, 3, 4).astype(np.float32) * 2,
            (-1e-3, 1e-3))]
    if quantize:
        func_args = [sign, n, delta, quantize, ste_fine_grained]
        function_tester(rng,
                        F.fixed_point_quantize,
                        ref_fixed_point_quantize,
                        inputs,
                        func_args=func_args,
                        atol_b=1e-3, backward=[True],
                        ctx=ctx, func_name=func_name,
                        ref_grad=ref_grad_fixed_point_quantize)
    else:  # No quantize
        for i in inputs:
            v = nn.Variable(i.shape)
            v.d = i
            v.g = np.random.randn(*i.shape)
            o = F.fixed_point_quantize(v)
            o.forward()
            o.backward()
            np.allclose(v.d, o.d)
            np.allclose(v.g, o.g)
