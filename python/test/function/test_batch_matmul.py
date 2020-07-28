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

ctxs = list_context('BatchMatmul')


def ref_batch_matmul(a, b, ta, tb):
    batch_dims_a = a.shape[:-2]
    batch_dims_b = b.shape[:-2]
    if batch_dims_a != batch_dims_b:
        batch_dims = tuple([max(da, db)
                            for da, db in zip(batch_dims_a, batch_dims_b)])
        a = np.broadcast_to(a, batch_dims + a.shape[-2:])
        b = np.broadcast_to(b, batch_dims + b.shape[-2:])
    a = np.transpose(a, [i for i in range(a.ndim - 2)] +
                     [a.ndim - 1, a.ndim - 2]) if ta else a
    b = np.transpose(b, [i for i in range(b.ndim - 2)] +
                     [b.ndim - 1, b.ndim - 2]) if tb else b
    y = np.matmul(a, b)
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("reduce_dim", [1, 5])
@pytest.mark.parametrize("row_a", [1, 5])
@pytest.mark.parametrize("col_b", [1, 5])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("batch_dims_a, batch_dims_b", ([
   # ((2, 2, 4), (2, 4)),  # this pattern is no longer supported since the batch dimensions have a meaning.
   # ((1,), tuple())      # this pattern is no longer supported since we expect > 3dims
   ((2, 3), (2, 3)),
   ((2, 3), (2, 1)),
   ((1, 3), (2, 3)),
   ((1, 3), (2, 1)),
   ((1, 1), (1, 1)),
]))
def test_batch_matmul_forward_backward(seed, reduce_dim, row_a, col_b, transpose_a, transpose_b, batch_dims_a, batch_dims_b, ctx, func_name):

    from nbla_test_utils import function_tester
    if transpose_a:
        shape_a = (reduce_dim, row_a)
    else:
        shape_a = (row_a, reduce_dim)
    if transpose_b:
        shape_b = (col_b, reduce_dim)
    else:
        shape_b = (reduce_dim, col_b)
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


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("reduce_dim", [1, 5])
@pytest.mark.parametrize("row_a", [1, 5])
@pytest.mark.parametrize("col_b", [1, 5])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("batch_dims_a, batch_dims_b", ([
   # ((2, 2, 4), (2, 4)),  # this pattern is no longer supported since the batch dimensions have a meaning.
   # ((1,), tuple())      # this pattern is no longer supported since we expect > 3dims
   ((2, 3), (2, 3)),
   ((2, 3), (2, 1)),
   ((1, 3), (2, 3)),
   ((1, 3), (2, 1)),
   ((1, 1), (1, 1)),
]))
def test_batch_matmul_double_backward(seed, reduce_dim, row_a, col_b, transpose_a, transpose_b, batch_dims_a, batch_dims_b, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    if transpose_a:
        shape_a = (reduce_dim, row_a)
    else:
        shape_a = (row_a, reduce_dim)
    if transpose_b:
        shape_b = (col_b, reduce_dim)
    else:
        shape_b = (reduce_dim, col_b)
    shape_a = batch_dims_a + shape_a
    shape_b = batch_dims_b + shape_b

    rng = np.random.RandomState(seed)
    # Input
    inputs = [
        rng.randn(*shape_a).astype(np.float32),
        rng.randn(*shape_b).astype(np.float32),
    ]
    backward_function_tester(rng, F.batch_matmul, inputs, func_args=[transpose_a, transpose_b],
                             atol_accum=1e-1, dstep=1e-3, ctx=ctx)
