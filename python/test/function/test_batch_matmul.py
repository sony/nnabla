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

ctxs = list_context('BatchMatmul')


def ref_batch_matmul(a, b, ta, tb):
    ra, ca = a.shape[-2:]
    rb, cb = b.shape[-2:]
    ry = ca if ta else ra
    cy = rb if tb else cb
    batch_dims = a.shape[:-2]
    samples = int(np.prod(batch_dims))
    a = a.reshape(samples, ra, ca)
    b = b.reshape(samples, rb, cb)
    yy = np.zeros((samples, ry, cy), dtype=np.float32)
    for i, (sa, sb) in enumerate(zip(a, b)):
        if ta:
            sa = sa.T
        if tb:
            sb = sb.T
        yy[i, ...] = np.dot(sa, sb)
    y = yy.reshape(batch_dims + (ry, cy))
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("reduc_dim", [1, 5])
@pytest.mark.parametrize("row_a", [1, 5])
@pytest.mark.parametrize("col_b", [1, 5])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("batch_dims_a, batch_dims_b", ([((2, 2, 2), (2, 4)), ((1,), tuple())]))
def test_batch_matmul_forward_backward(seed, reduc_dim, row_a, col_b, transpose_a, transpose_b, batch_dims_a, batch_dims_b, ctx, func_name):

    from nbla_test_utils import function_tester
    if transpose_a:
        shape_a = (reduc_dim, row_a)
    else:
        shape_a = (row_a, reduc_dim)
    if transpose_b:
        shape_b = (col_b, reduc_dim)
    else:
        shape_b = (reduc_dim, col_b)
    shape_a = batch_dims_a + shape_a
    shape_b = batch_dims_b + shape_b

    rng = np.random.RandomState(seed)
    # Input
    inputs = [
        rng.randn(*shape_a).astype(np.float32),
        rng.randn(*shape_b).astype(np.float32),
    ]
    function_tester(rng, F.batch_matmul, ref_batch_matmul, inputs, func_args=[transpose_a, transpose_b],
                    atol_b=2e-2, dstep=1e-3, ctx=ctx, func_name=func_name)
