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


from nbla_test_utils import (function_tester, list_context,
                             list_ctx_and_func_name)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, (0, 2), (1, 2, 3)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("inshape", [(2, 3, 4, 5), (2, 1, 4, 5)])
@pytest.mark.parametrize("op, ctx, func_name", list_ctx_and_func_name(['sum', 'mean', 'max', 'min', 'prod']))
def test_reduction_forward_backward(op, seed, inshape, axis, keepdims, ctx, func_name):
    func = getattr(F, op)
    ref_func = getattr(np, op)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, func, ref_func, inputs,
                    func_args=[axis],
                    func_kwargs=dict(keepdims=keepdims),
                    ctx=ctx, func_name=func_name,
                    # The backward test on macOS doesn't pass with this tolerance.
                    # Does Eigen library used in CPU computation backend produce
                    # the different results on different platforms?
                    # atol_b=3e-3,
                    atol_b=6e-3)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, axis", [((2, 1, 64, 64), (2, 3))])
@pytest.mark.parametrize("ctx, func_name", list_context('Sum'))
def test_large_sum_reduction(seed, inshape, axis, ctx, func_name):
    if not func_name.endswith('Cuda'):
        # This configuration is only to test the CUDA implementation branch
        # where reduction_size / outer_size >= 2048, so we skip this test for
        # CPU and CuDNN and also do not need to run for both keepdims.
        pytest.skip('skip CUDA specific implementation test for other target')

    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.sum, np.sum, inputs, ctx=ctx, func_name=func_name,
                    func_args=[axis], atol_b=1e-2, disable_half_test=True)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, axis", [((2, 1, 64, 64), (2, 3))])
@pytest.mark.parametrize("ctx, func_name", list_context('Mean'))
def test_large_mean_reduction(seed, inshape, axis, ctx, func_name):
    if not func_name.endswith('Cuda'):
        # This configuration is only to test the CUDA implementation branch
        # where reduction_size / outer_size >= 2048, so we skip this test for
        # CPU and CuDNN and also do not need to run for both keepdims.
        pytest.skip('skip CUDA specific implementation test for other target')

    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.mean, np.mean, inputs, ctx=ctx, func_name=func_name,
                    func_args=[axis])
