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

ctxs = list_context('GELU')


def ref_gelu(x):
    return x/2*(1+np.tanh(np.sqrt(2/np.pi)*(x + 0.044715 * np.power(x, 3))))


# def ref_gelu_backward(x, dx, **kw):
#    return np.array(np.array(0.5 + (0.398942*x + 0.0535161 * np.power(x, 3)) * np.power(1 / np.cosh(0.797885*x + 0.0356774 * np.power(x, 3)), 2) + 0.5*np.tanh(0.797885*x + 0.0356774*np.power(x, 3))).flat)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313, 999])
def test_gelu_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    function_tester(rng, F.gelu, ref_gelu, inputs,
                    ctx=ctx, func_name=func_name,
                    atol_b=1e-3, atol_accum=1e-3)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313, 999])
def test_gelu_double_backward(seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32)]
    backward_function_tester(rng, F.gelu, inputs,
                             ctx=ctx,
                             atol_accum=5e-3)
