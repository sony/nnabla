# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

ctxs = list_context('ScatterAdd')


def scatter_add(x0, indices, x1, axis):
    output = np.copy(x0)
    if x0.ndim == 2:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                if axis == 0 or axis == -2:
                    output[indices[i][j]][j] += x1[i][j]
                elif axis == 1 or axis == -1:
                    output[i][indices[i][j]] += x1[i][j]
    elif x0.ndim == 3:
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    if axis == 0 or axis == -3:
                        output[indices[i][j][k]][j][k] += x1[i][j][k]
                    elif axis == 1 or axis == -2:
                        output[i][indices[i][j][k]][k] += x1[i][j][k]
                    elif axis == 2 or axis == -1:
                        output[i][j][indices[i][j][k]] += x1[i][j][k]
    return output


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x0_shape, indices_shape, x1_shape, axis",
                         [((1, 1), (1, 1), (1, 1), 0),
                          ((2, 2), (1, 1), (3, 3), -2),
                          ((3, 3), (1, 1), (2, 2), 1),
                          ((3, 3), (1, 1), (3, 3), -1),
                          ((1, 10), (1, 10), (1, 10), 0),
                          ((2, 12), (1, 10), (3, 11), -2),
                          ((3, 11), (1, 10), (2, 12), 1),
                          ((2, 11), (1, 10), (2, 12), -1),
                          ((10, 10), (10, 10), (10, 10), 0),
                          ((12, 11), (10, 10), (11, 12), 1),
                          ((12, 12), (10, 10), (12, 11), -1),
                          ((5, 4, 3), (5, 4, 3), (5, 4, 3), 0),
                          ((6, 5, 4), (5, 4, 3), (7, 8, 9), -3),
                          ((5, 5, 5), (3, 4, 5), (7, 8, 9), 1),
                          ((4, 5, 6), (3, 4, 5), (3, 4, 5), -2),
                          ((5, 7, 6), (3, 5, 4), (7, 9, 8), 2),
                          ((3, 5, 4), (3, 5, 4), (7, 7, 7), -1)])
def test_forward_backward(seed, x0_shape, indices_shape, x1_shape, axis, ctx, func_name):
    rng = np.random.RandomState(seed)
    x0 = rng.randn(*x0_shape).astype(np.float32)
    indices = rng.randint(low=0, high=min(indices_shape),
                          size=np.prod(indices_shape))
    indices = np.reshape(indices, newshape=indices_shape)
    x1 = rng.randn(*x1_shape).astype(np.float32)

    inputs = [x0, indices, x1]
    function_tester(rng, F.scatter_add, scatter_add, inputs,
                    func_name=func_name, func_args=[
                        axis], ctx=ctx, backward=[True, False, True],
                    atol_b=3e-3)
