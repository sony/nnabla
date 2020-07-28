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

ctxs = list_context('ELU')


def ref_elu(x, alpha):
    return np.maximum(x, 0) + alpha * (np.exp(np.minimum(x, 0)) - 1)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("seed", [313])
def test_elu_forward_backward(seed, alpha, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]
    function_tester(rng, F.elu, ref_elu, inputs, func_args=[alpha],
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("seed", [313])
def test_elu_double_backward(seed, alpha, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3).astype(np.float32) * 2]
    backward_function_tester(rng, F.elu,
                             inputs=inputs,
                             func_args=[alpha], func_kwargs={},
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
