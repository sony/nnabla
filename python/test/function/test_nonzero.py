# Copyright 2022 Sony Corporation.
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
from nnabla.testing import assert_allclose

ctxs = list_context('NonZero')


def ref_nonzero(x):
    return np.array(np.nonzero(x), dtype=np.uint64)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [
    (1,),
    (2,),
    (2, 3),
    (2, 3, 4),
    (2, 3, 4, 5),
    (2, 3, 4, 5, 6),
    (1, 1),
    (1, 10000),
    (100000, 1),
])
@pytest.mark.parametrize("zero_rate", [0.0, 0.5, 1.0])
def test_nonzero_forward(seed, inshape, zero_rate, ctx, func_name):
    rng = np.random.RandomState(seed)
    input = rng.randn(*inshape).astype(dtype=np.float32)
    cond = (rng.rand(*inshape) <= zero_rate)
    input = np.where(cond, np.zeros_like(input), input)

    vinput = nn.Variable.from_numpy_array(input)
    with nn.context_scope(ctx), nn.auto_forward():
        o = F.nonzero(vinput)
    r = ref_nonzero(input)
    assert_allclose(o.d, r)
    assert func_name == o.parent.name
