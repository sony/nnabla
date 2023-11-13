# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

ctxs = list_context('Trilu')


def ref_trilu(x, k=0, upper=True):
    if upper:
        return np.triu(x, k)
    else:
        return np.tril(x, k)


shapes = [
    (1, 3, 3),
    (2, 4, 3),
    (2, 2, 3, 4),
]


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("k", list(range(-4, 4)))
@pytest.mark.parametrize("upper", [True, False])
def test_trilu_forward_backward(seed, ctx, func_name, shape, k, upper):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(*shape).astype(np.float32) * 2,
            (-1e-3, 1e-3))]
    function_tester(rng, F.trilu, ref_trilu, inputs, atol_b=1e-2, dstep=1e-3,
                    ctx=ctx, func_name=func_name, func_args=[k, upper])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("k", list(range(-4, 4)))
@pytest.mark.parametrize("upper", [True, False])
def test_trilu_double_backward(seed, ctx, func_name, shape, k, upper):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [
        cap_ignore_region(
            rng.randn(*shape).astype(np.float32) * 2,
            (-1e-3, 1e-3))]
    backward_function_tester(rng, F.trilu,
                             inputs=inputs,
                             func_args=[k, upper], func_kwargs={},
                             atol_accum=1e-3,
                             dstep=1e-3,
                             backward_b=[True, False],
                             ctx=ctx)
