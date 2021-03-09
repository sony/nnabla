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

ctxs = list_context('Where')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_where_forward_backward(seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inshape = (2, 3, 4)
    inputs = [
        (rng.rand(*inshape) > 0.5).astype(np.float32),
        rng.randn(*inshape),
        rng.randn(*inshape),
    ]
    function_tester(rng, F.where, np.where, inputs,
                    backward=[False, True, True],
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_where_double_backward(seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inshape = (2, 2)
    inputs = [
        (rng.rand(*inshape) > 0.5).astype(np.float32),
        rng.randn(*inshape),
        rng.randn(*inshape),
    ]
    backward_function_tester(rng, F.where, inputs,
                             backward=[False, True, True], dstep=1e3,
                             ctx=ctx)
