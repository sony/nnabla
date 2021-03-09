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

ctxs = list_context('SELU')


def ref_selu(x, scale, alpha):
    return np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("scale", [1.050700987355480, 1.0, 0.5, 0.0])
@pytest.mark.parametrize("alpha", [1.673263242354377, 1.0, 0.5, 0.0])
@pytest.mark.parametrize("seed", [313])
def test_selu_forward_backward(seed, scale, alpha, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]
    function_tester(rng, F.selu, ref_selu, inputs, func_args=[scale, alpha],
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("scale", [1.050700987355480, 1.0, 0.5, 0.0])
@pytest.mark.parametrize("alpha", [1.673263242354377, 1.0, 0.5, 0.0])
def test_selu_double_backward(seed, scale, alpha, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]
    backward_function_tester(rng, F.selu,
                             inputs=inputs,
                             func_args=[scale, alpha], func_kwargs={},
                             atol_accum=1e-2,
                             dstep=1e-3,
                             ctx=ctx)
