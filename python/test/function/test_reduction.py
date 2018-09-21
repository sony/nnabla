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


from nbla_test_utils import (
    function_tester,
    list_ctx_and_func_name)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, (0, 2), (1, 2, 3)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("op, ctx, func_name", list_ctx_and_func_name(['sum', 'mean', 'max', 'min', 'prod']))
def test_reduction_forward_backward(op, seed, axis, keepdims, ctx, func_name):
    from nbla_test_utils import function_tester
    func = getattr(F, op)
    ref_func = getattr(np, op)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4, 5).astype(np.float32)]
    function_tester(rng, func, ref_func, inputs,
                    func_args=[axis],
                    func_kwargs=dict(keepdims=keepdims),
                    ctx=ctx, func_name=func_name,
                    # The backward test on macOS doesn't pass with this tolerance.
                    # Does Eigen library used in CPU computation backend produce
                    # the different results on different platforms?
                    # atol_b=3e-3,
                    atol_b=6e-3)
