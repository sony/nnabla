# Copyright 2022 Sony Group Corporation.
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

ctxs = list_context('Shape')


def ref_shape(x, start, end):
    if start is None:
        start = 0
    if end is None:
        end = 100
    return x.shape[start:end],


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", [(3, 4, 5), (7, 8, 9, 10)])
@pytest.mark.parametrize("start, end", [(0, 100), (None, None), (None, -1), (0, None)])
def test_constant_forward(seed, shape, start, end, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [np.random.random(shape)]
    function_tester(rng, F.shape, ref_shape, inputs, func_args=[start, end],
                    ctx=ctx, func_name=func_name, backward=None)


# No need to test backward_function
