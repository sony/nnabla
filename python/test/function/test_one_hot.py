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

ctxs = list_context('OneHot')


def ref_one_hot(x, shape):
    result = np.zeros((x.shape[0],) + shape)
    result[np.arange(x.shape[0]), x.flatten()] = 1
    return result


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(100, 1)])
@pytest.mark.parametrize("shape", [(10,)])
def test_one_hot_forward(seed, inshape, shape, ctx, func_name):
    rng = np.random.RandomState(seed)
    # Input
    input = rng.randint(0, shape[0], size=inshape)
    vinput = nn.Variable(input.shape, need_grad=False)
    vinput.d = input
    with nn.context_scope(ctx), nn.auto_forward():
        o = F.one_hot(vinput, shape)
    r = ref_one_hot(input, shape)
    assert np.allclose(o.d, r)
    assert func_name == o.parent.name
