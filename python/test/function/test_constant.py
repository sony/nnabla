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

ctxs = list_context('Constant')


def ref_constant(val, shape):
    return np.ones(shape, dtype=np.float32) * val


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("value", [0., 1.5, -2., 1000.])
@pytest.mark.parametrize("shape", [[], [5], [2, 3, 4]])
def test_constant_forward(value, shape, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = None
    inputs = []
    function_tester(rng, F.constant, ref_constant, inputs, func_args=[value, shape],
                    ctx=ctx, func_name=func_name, backward=[])


# No need to test backward_function
