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

ctxs = list_context('Embed')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape_x", [(10,), (2, 8), (2, 3, 4), (2, 2, 3, 4)])
@pytest.mark.parametrize("shape_w", [(5, 3), (4, 3, 4), (6, 2, 2, 3)])
def test_embed_forward_backward(seed, shape_x, shape_w, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    n_class = shape_w[0]
    x = rng.randint(0, n_class - 1, shape_x).astype(np.int32)
    w = rng.randn(*shape_w).astype(np.float32)
    inputs = [x, w]
    function_tester(rng, F.embed, lambda x, w: w[x], inputs,
                    ctx=ctx, func_name=func_name, atol_b=1e-2,
                    backward=[False, True])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape_x", [(10,), (2, 8), (2, 3, 4), (2, 2, 3, 4)])
@pytest.mark.parametrize("shape_w", [(5, 3), (4, 3, 4), (6, 2, 2, 3)])
def test_embed_double_backward(seed, shape_x, shape_w, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    n_class = shape_w[0]
    x = rng.randint(0, n_class - 1, shape_x).astype(np.int32)
    w = rng.randn(*shape_w).astype(np.float32)
    inputs = [x, w]
    backward_function_tester(rng, F.embed, None, inputs,
                             ctx=ctx, func_name=func_name, atol_b=6e-2, atol_accum=6e-2,
                             dstep=1e-3,
                             backward=[False, True])
