# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('ScatterNd')


def scatter_nd(data, indices, shape):
    indices = indices.tolist() if isinstance(indices, np.ndarray) else indices
    result = np.zeros(shape, dtype=data.dtype)
    result[indices] = data
    return result


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ishape, index, oshape", [
    ((2,), [[1, 3]], (10,)),
    ((2,), [[-1, -3]], (10,)),
    ((3,), [[1, 1, 0], [0, 1, 0]], (2, 2)),
    ((4,), [[4, 3, 1, 7]], (8,)),
    ((2, 4), [[0, 1], [2, 3]], (4, 4, 4)),
    ((2, 4, 4), [[0, 2]], (4, 4, 4)),
    ((2, 2, 2), [[0, 1], [1, 1]], (2, 2, 2, 2)),
])
def test_forward_backward(seed, ishape, index, oshape, ctx, func_name):
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*ishape).astype(np.float32), np.array(index)]
    function_tester(rng, F.scatter_nd, scatter_nd, inputs, func_name=func_name,
                    func_args=[oshape], ctx=ctx, backward=[True, False])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ishape, index, oshape", [
    ((2,), [[1, 3]], (10,)),
    ((2,), [[-1, -3]], (10,)),
    ((3,), [[1, 1, 0], [0, 1, 0]], (2, 2)),
    ((4,), [[4, 3, 1, 7]], (8,)),
    ((2, 4), [[0, 1], [2, 3]], (4, 4, 4)),
    ((2, 4, 4), [[0, 2]], (4, 4, 4)),
    ((2, 2, 2), [[0, 1], [1, 1]], (2, 2, 2, 2)),
])
def test_double_backward(seed, ishape, index, oshape, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*ishape).astype(np.float32), np.array(index) * 0.1]
    backward_function_tester(rng, F.scatter_nd, inputs,
                             func_args=[oshape], ctx=ctx, backward=[
                                 True, False],
                             atol_accum=1e-2)
