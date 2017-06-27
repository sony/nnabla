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

ctxs = list_context('Slice')


def ref_slice(x, start, stop, step):
    s = [slice(start[axis], stop[axis], step[axis])
         for axis in range(len(start))]
    return x[s]


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, start, stop, step",
                         [((2, 2), (0, 0), (2, 2), (1, 1)), ((6, 7, 8), (1, 2, 3), (5, 4, 8), (1, 1, 2)), ((6, 7, 6, 5), (4, 3, 2, 1), (5, 6, 5, 4), (1, 2, 3, 4)), ((7, 6, 5, 4, 3), (5, 4, 3, 2, 1), (6, 6, 5, 4, 2), (1, 2, 3, 2, 1))
                          ])
def test_slice_forward_backward(seed, inshape, start, stop, step, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    rng = np.random.RandomState(seed)
    x = rng.randn(*inshape).astype(np.float32)
    function_tester(rng, F.slice, ref_slice, [x],
                    func_args=[start, stop,
                               step], ctx=ctx, func_name=func_name,
                    atol_f=1e-4, atol_b=1e-2)
