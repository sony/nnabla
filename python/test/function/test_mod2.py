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

ctxs = list_context('Mod2')


def ref_mod2(x0, x1, fmod):
    if x0.dtype == np.float32 or fmod == True:
        return np.fmod(x0, x1)
    else:
        return np.mod(x0, x1)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("x0_shape, x1_shape", [
    ((2, 3, 4), (2, 3, 4)),
    ((2, 3, 4), (1, 1, 1)),
    ((1, 1, 1), (2, 3, 4)),
])
@pytest.mark.parametrize('fmod', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.int32])
@pytest.mark.parametrize("seed", [313])
def test_mod2_forward(seed, x0_shape, x1_shape, fmod, dtype, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    if dtype == np.float32:
        inputs = [rng.randn(*x0_shape).astype(dtype),
                  rng.randn(*x1_shape).astype(dtype)]
    else:
        inputs = [rng.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, x0_shape).astype(dtype),
                  rng.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, x1_shape).astype(dtype)]
    backward = [False, False]
    func_args = [fmod]
    function_tester(rng, F.mod2, ref_mod2, inputs,
                    func_name=func_name, func_args=func_args,
                    atol_f=0, ctx=ctx, backward=backward)
