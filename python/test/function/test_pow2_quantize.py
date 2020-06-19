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

ctxs = list_context('Pow2Quantize')


def ref_pow2_quantize(x, sign, with_zero, n, m, quantize, ste_fine_grained):
    assert (n > 0)

    n_ = n - 1 if sign else n
    n_ = n_ - 1 if with_zero else n_
    ref_p_max = 2 ** m
    ref_p_min = 2 ** (m - ((1 << n_) - 1))
    ref_pruning_threshold = ref_p_min * (2. ** -0.5)
    if quantize:
        x_q = 2. ** np.round(np.log2(np.abs(x)))
        x_q[np.where(x_q > ref_p_max)] = ref_p_max
        if with_zero:
            x_q_ = x_q.copy()
            x_q[np.where(x_q_ < ref_p_min)] = ref_p_min
            x_q[np.where(np.abs(x) < ref_pruning_threshold)] = 0.
        if not with_zero:
            x_q[np.where(x_q < ref_p_min)] = ref_p_min
        if sign:
            x_q = np.sign(x) * x_q
        else:
            if with_zero:
                x_q[np.where(np.sign(x) < 0.)] = 0.
            else:
                x_q[np.where(np.sign(x) < 0.)] = ref_p_min
    else:
        x_q = x
    return x_q


def ref_grad_pow2_quantize(x, dy, sign,
                           with_zero, n, m, quantize, ste_fine_grained, **kw):
    if not ste_fine_grained:
        return dy.flatten()

    n_ = n - 1 if sign else n
    n_ = n_ - 1 if with_zero else n_
    ref_p_max = 2 ** m
    ref_p_min = 2 ** (m - ((1 << n_) - 1))
    ref_pruning_threshold = ref_p_min * (2. ** -0.5)
    x_q = 2. ** np.round(np.log2(np.abs(x)))
    c = np.ones_like(x)
    c[np.where(x_q > ref_p_max)] = 0.

    if not sign:
        c[np.where(np.sign(x) < 0.)] = 0.

    return (c * dy).flatten()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("sign", [False, True])
@pytest.mark.parametrize("with_zero", [False, True])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("ste_fine_grained", [False, True])
def test_pow2_quantize_forward_backward(seed,
                                        sign, with_zero, n, m, quantize,
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
        func_args = [sign, with_zero, n, m, quantize, ste_fine_grained]
        function_tester(rng,
                        F.pow2_quantize,
                        ref_pow2_quantize,
                        inputs,
                        func_args=func_args,
                        atol_b=1e-3, backward=[True],
                        ctx=ctx, func_name=func_name,
                        ref_grad=ref_grad_pow2_quantize)

    else:  # No quantize
        for i in inputs:
            v = nn.Variable(i.shape)
            v.d = i
            v.g = np.random.randn(*i.shape)
            o = F.pow2_quantize(v)
            o.forward()
            o.backward()
            np.allclose(v.d, o.d)
            np.allclose(v.g, o.g)
