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
from nbla_test_utils import list_ctx_and_func_name2


@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name2([('isnan', 'IsNaN'), ('isinf', 'IsInf')]))
@pytest.mark.parametrize("seed", [313])
def test_isnan_isinf_forward(seed, fname, ctx, func_name):
    from nbla_test_utils import function_tester
    np_fun = getattr(np, fname)
    nn_fun = getattr(F, fname)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    inputs[0][rng.rand(*inputs[0].shape) > 0.5] = getattr(np,
                                                          fname.replace('is', ''))

    def ref_forward(x):
        return np_fun(x).astype(np.int32).astype(np.float32)
    function_tester(rng, nn_fun, ref_forward, inputs, backward=[False],
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("fname, ctx, func_name",
                         list_ctx_and_func_name2([('reset_nan', 'ResetNaN'), ('reset_inf', 'ResetInf')]))
@pytest.mark.parametrize("val", [0, -1])
@pytest.mark.parametrize("seed", [313])
def test_reset_nan_reset_inf_forward_backward(seed, val, fname, ctx, func_name):
    from nbla_test_utils import function_tester
    np_fun = getattr(np, fname.replace('reset_', 'is'))
    nn_fun = getattr(F, fname)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    inputs[0][rng.rand(*inputs[0].shape) > 0.5] = getattr(np,
                                                          fname.replace('reset_', ''))

    def ref_forward(x, val):
        y = x.copy()
        y[np_fun(x)] = val
        return y

    def ref_backward(x, dy, val, **kw):
        dx = dy.copy()
        dx[np_fun(x)] = 0
        return dx.flatten()

    function_tester(rng, nn_fun, ref_forward, inputs, func_args=[val],
                    ref_grad=ref_backward,
                    ctx=ctx, func_name=func_name)
