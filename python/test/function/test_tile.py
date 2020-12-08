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

ctxs = list_context('Tile')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("inshape", [(12,), (3, 5), (2, 3, 4), (4, 3, 8, 8)])
@pytest.mark.parametrize("reps", [(2,), (3, 2), (3, 2, 3), (3, 2, 3, 5)])
def test_tile_forward_backward(inshape, reps, seed, ctx, func_name):
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.tile, np.tile, inputs, ctx=ctx,
                    func_name=func_name, func_args=[reps], atol_b=1e-2,
                    disable_half_test=False, backward=[False])
