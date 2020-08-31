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

ctxs = list_context('GatherNd')


def gather_nd(data, index):
    index_ = index.reshape(index.shape[0], -1)  # flatten inner dims
    index_ = (idx + (Ellipsis,) for idx in zip(*index_))
    result = np.vstack(data[idx] for idx in index_)
    result = result.reshape(*(index.shape[1:] + data.shape[index.shape[0]:]))
    return result


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ishape, index", [
    ([10], [[0]]),
    ([10], [[1, 5, 8]]),
    ([10], [[-1, -5, -8]]),
    ([3, 4], [[0]]),
    ([3, 4], [[0], [0]]),
    ([3, 4], [[0, 1], [0, 2]]),
    ([3, 4], [[0, -1], [0, -2]]),
    ([2, 3, 4], [[0]]),
    ([2, 3, 4], [[0], [1]]),
    ([2, 3, 4], [[0], [1], [2]]),
    ([2, 3, 4], [[0, 1]]),
    ([2, 3, 4], [[0, 1], [1, 2]]),
    ([2, 3, 4], [[0, 1], [1, 2], [1, 0]]),
    ([4, 4, 4, 4], [[[0, 1], [2, 3]], [[0, 1], [2, 3]]]),
])
def test_forward_backward(seed, ishape, index, ctx, func_name):
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*ishape).astype(np.float32), np.array(index)]
    function_tester(rng, F.gather_nd, gather_nd, inputs, func_name=func_name,
                    ctx=ctx, backward=[True, False])
