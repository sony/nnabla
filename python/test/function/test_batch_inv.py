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


ctxs = list_context('BatchInv')


def ref_inv(x):
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        y[i] = np.linalg.inv(x[i])
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
def test_batch_inv_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # input must be batched square matrix
    inputs = [np.clip(rng.randn(2, 3, 3).astype(np.float32), -0.9, 0.9)]
    function_tester(rng, F.batch_inv, ref_inv, inputs, ctx=ctx,
                    func_name=func_name, atol_b=2e-2, dstep=1e-4,
                    disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
def test_batch_inv_double_backward(seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    # input must be batched square matrix
    inputs = [np.clip(rng.randn(2, 3, 3).astype(np.float32), -0.9, 0.9) * 10]
    backward_function_tester(rng, F.batch_inv, inputs,
                             ctx=ctx, atol_accum=1e-2)
