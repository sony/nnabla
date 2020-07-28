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

ctxs = list_context('KLMultinomial')


def ref_kl_multinomial(p, q, base_axis):
    kl = np.sum(p * (np.log(p + 1.0e-8) - np.log(q + 1.0e-8)),
                axis=tuple(range(base_axis, p.ndim)))
    return kl.reshape(kl.shape + (1,))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("base_axis, shape", [(1, (3, 6)), (1, (5, 8, 7)), (2, (4, 7, 9))])
def test_kl_multinomial_forward_backward(seed, ctx, base_axis, shape, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    input0 = 1 + rng.rand(*(shape)).astype(np.float32)
    input1 = 1 + rng.rand(*(shape)).astype(np.float32)
    input0 = input0 / np.sum(input0, axis=1, keepdims=True)
    input1 = input1 / np.sum(input1, axis=1, keepdims=True)
    inputs = [input0, input1]
    function_tester(rng, F.kl_multinomial, ref_kl_multinomial, inputs, func_args=[base_axis],
                    atol_f=1e-6, atol_b=1e-2, dstep=1e-4,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("base_axis, shape", [(1, (3, 6)), (1, (5, 8, 7)), (2, (4, 7, 9))])
def test_kl_multinomial_double_backward(seed, ctx, base_axis, shape, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    input0 = 1 + rng.rand(*(shape)).astype(np.float32)
    input1 = 1 + rng.rand(*(shape)).astype(np.float32)
    input0 = input0 / np.sum(input0, axis=1, keepdims=True)
    input1 = input1 / np.sum(input1, axis=1, keepdims=True)
    inputs = [input0, input1]
    backward_function_tester(rng, F.kl_multinomial, inputs, func_args=[base_axis],
                             atol_f=1e-6, dstep=1e-3,
                             ctx=ctx, skip_backward_check=True)
