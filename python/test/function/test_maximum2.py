# Copyright 2019,2020,2021 Sony Corporation.
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

"""
This is still left unlike other *2 operation (sub2, mul2, ...) because
it has cudnn implementation.
"""
import pytest
import numpy as np
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Maximum2')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_maximum2_double_backward(seed, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3).astype(np.float32),
              rng.randn(2, 3).astype(np.float32)]
    backward_function_tester(rng, F.maximum2,
                             inputs=inputs,
                             func_args=[], func_kwargs={},
                             atol_accum=1e-3,
                             dstep=1e-3,
                             ctx=ctx)
