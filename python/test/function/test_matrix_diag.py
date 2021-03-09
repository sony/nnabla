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

ctxs = list_context('MatrixDiag')

# Test is not general implementation of matrix diagonal.


def ref_matrix_diag(x, ):
    y = np.zeros(x.shape + (x.shape[-1], ))
    if x.ndim == 2:
        for b in range(x.shape[0]):
            y[b, :, :] = np.diag(x[b, :])
    if x.ndim == 3:
        for t in range(x.shape[0]):
            for b in range(x.shape[1]):
                y[t, b, :, :] = np.diag(x[t, b, :])
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])  # (b, d) or (t, b, d)
def test_matrix_diag_forward_backward(seed, ctx, func_name, shape):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*shape).astype(np.float32) * 0.1]
    function_tester(rng, F.matrix_diag, ref_matrix_diag, inputs, func_args=[],
                    atol_b=1e-3, ctx=ctx, func_name=func_name,)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])  # (b, d) or (t, b, d)
def test_matrix_diag_double_backward(seed, ctx, func_name, shape):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*shape).astype(np.float32) * 0.1]
    backward_function_tester(rng, F.matrix_diag, inputs, func_args=[], ctx=ctx)
