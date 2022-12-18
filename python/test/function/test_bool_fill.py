# Copyright 2021 Sony Group Corporation.
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

ctxs = list_context('BoolFill')


def ref_bool_fill(data, mask, value):
    data0 = np.copy(data)
    bmask = np.broadcast_to(mask, data.shape)
    data0[bmask.astype(bool)] = value
    return data0


@pytest.mark.parametrize("inf_or_nan", [0, np.inf, np.nan])
@pytest.mark.parametrize("value", [1, -1.5])
@pytest.mark.parametrize("dshape, mshape",
                         [((2, 3), (2, 3)),
                          ((5, ), (5, )),
                          ((2, 3), (3, )),
                          ((2, 3, 2, 2), (1, 2, 1))
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_fill_forward_backward(seed, ctx, func_name, dshape, mshape, value, inf_or_nan):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    data = rng.randn(*dshape).astype(np.float32)
    mask = rng.randint(0, 2, mshape)
    backward = [True, False]
    if inf_or_nan != 0:
        bmask = np.broadcast_to(mask, data.shape)
        data[bmask.astype(bool)] = inf_or_nan
        # we can not compute the numerical gradients
        backward = [False, False]
    inputs = [data, mask]

    function_tester(rng, F.bool_fill, ref_bool_fill, inputs,
                    ctx=ctx, func_name=func_name, func_args=[value],
                    backward=backward)


@pytest.mark.parametrize("inf_or_nan", [0, np.inf, np.nan])
@pytest.mark.parametrize("value", [1, -1.5])
@pytest.mark.parametrize("dshape, mshape",
                         [((2, 3), (2, 3)),
                          ((5, ), (5, )),
                          ((2, 3), (3, )),
                          ((2, 3, 2, 2), (1, 2, 1))
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_fill_double_backward(seed, ctx, func_name, dshape, mshape, value, inf_or_nan):
    from nbla_test_utils import backward_function_tester

    rng = np.random.RandomState(seed)
    data = rng.randn(*dshape).astype(np.float32)
    mask = rng.randint(0, 2, mshape)
    backward = [True, False]
    backward_b = [True, True, False]
    if inf_or_nan != 0:
        bmask = np.broadcast_to(mask, data.shape)
        data[bmask.astype(bool)] = inf_or_nan
        backward = [True, False]
        # we can not compute the numerical gradients
        backward_b = [False, False, False]
    inputs = [data, mask]

    backward_function_tester(rng, F.bool_fill, inputs, ctx=ctx,
                             backward=backward,
                             backward_b=backward_b)
