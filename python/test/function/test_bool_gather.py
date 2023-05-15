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

ctxs = list_context('BoolGather')


def ref_bool_gather(gdata, mask):
    mask_bool = mask.astype(bool)
    return gdata[mask_bool]


@pytest.mark.parametrize("gshape, mask_shape",
                         [((2, 3, 2), (2, 3)),
                          ((3, 4), (3, 4)),
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_gather_forward_backward(seed, ctx, func_name, gshape, mask_shape):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    gdata = rng.randn(*gshape).astype(np.float32)
    mask = rng.randint(0, 2, size=mask_shape)
    inputs = [gdata, mask]

    function_tester(rng, F.bool_gather, ref_bool_gather, inputs,
                    ctx=ctx, func_name=func_name,
                    auto_forward=True,
                    backward=[True, False])


@pytest.mark.parametrize("gshape, mask_shape",
                         [((2, 3, 2), (2, 3)),
                          ((3, 4), (3, 4)),
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_gather_double_backward(seed, ctx, func_name, gshape, mask_shape):
    from nbla_test_utils import backward_function_tester

    rng = np.random.RandomState(seed)
    gdata = rng.randn(*gshape).astype(np.float32)
    mask = rng.randint(0, 2, size=mask_shape)
    inputs = [gdata, mask]

    backward_function_tester(rng, F.bool_gather, inputs, ctx=ctx,
                             backward=[True, False],
                             backward_b=[True, True, False],
                             auto_forward=True)


@pytest.mark.parametrize("gshape, mask_shape, reset_gshape, reset_mask_shape",
                         [((2, 3, 2), (2, 3), (2, 1, 2), (2, 1)),
                          ((3, 4), (3, 4), (4, 5), (4, 5)),
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_gather_forward_backward_with_reset(seed, ctx, func_name, gshape, mask_shape, reset_gshape,
                                                 reset_mask_shape):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    gdata = rng.randn(*gshape).astype(np.float32)
    reset_gdata = rng.randn(*reset_gshape).astype(np.float32)
    mask = rng.randint(0, 2, size=mask_shape)
    reset_mask = rng.randint(0, 2, size=reset_mask_shape)
    inputs = [gdata, mask]
    reset_inputs = [reset_gdata, reset_mask]
    function_tester(rng, F.bool_gather, ref_bool_gather, inputs,
                    ctx=ctx, func_name=func_name,
                    auto_forward=True,
                    backward=[True, False], reset_inputs=reset_inputs)
