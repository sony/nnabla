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
from nnabla.testing import assert_allclose

ctxs = list_context('OneHot')


def ref_one_hot(x, shape):
    result = np.zeros((x.shape[0],) + shape)
    for i in range(x.shape[0]):
        idx = (i, ) + tuple(x[i])
        result[idx] = 1
    return result


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(100, 1), (100, 2)])
@pytest.mark.parametrize("shape", [(10, ), (10, 8)])
def test_one_hot_forward(seed, inshape, shape, ctx, func_name):
    # Input
    input = np.zeros(inshape, dtype=int)
    rng = np.random.RandomState(seed)

    if len(shape) != inshape[-1]:
        # input inshape and shape don't match.
        with pytest.raises(RuntimeError):
            y = F.one_hot(nn.Variable(input.shape), shape)
    else:
        for i in range(inshape[-1]):
            input[:, i] = rng.randint(0, shape[i], size=inshape[0])
        vinput = nn.Variable(input.shape, need_grad=False)
        vinput.d = input

        with nn.context_scope(ctx), nn.auto_forward():
            o = F.one_hot(vinput, shape)
        r = ref_one_hot(input, shape)
        assert_allclose(o.d, r)
        assert func_name == o.parent.name
