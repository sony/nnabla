# Copyright 2018,2019,2020,2021 Sony Corporation.
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

ctxs = list_context('BitShift')


def ref_bit_shift(x, shift, direction):
    if direction == "LEFT":
        return x << shift
    elif direction == "RIGHT":
        return x >> shift
    else:
        raise ValueError("Invalid direction: {}".format(direction))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("x_shape, shift_shape", [
    ((2, 3, 4), (2, 3, 4)),
    ((2, 3, 4), (1, 1, 1)),
    ((1, 1, 1), (2, 3, 4)),
])
@pytest.mark.parametrize('direction', ("LEFT", "RIGHT"))
@pytest.mark.parametrize('dtype', (np.uint8, np.uint32))
@pytest.mark.parametrize("seed", [313])
def test_bit_shift_forward(seed, x_shape, shift_shape, direction, dtype, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # np.iinfo(dtype).max is divided by 2 to aviod out of bounds err
    inputs = [rng.randint(0, np.iinfo(dtype).max//2, x_shape).astype(dtype),
              rng.randint(0, 128, shift_shape).astype(dtype)]
    backward = [False, False]
    func_args = [direction]
    function_tester(rng, F.bit_shift, ref_bit_shift, inputs,
                    func_name=func_name, func_args=func_args, ctx=ctx, backward=backward)
