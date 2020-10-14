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

ctxs = list_context('WeightNormalization')


def ref_weight_normalization(w, g, dim=0, eps=1e-12):
    axes = tuple([a for a in range(w.ndim) if a != dim])
    n = (np.sum(w ** 2, axes, keepdims=True) + eps) ** (-0.5)
    rshape = [1 if i != dim else s for i, s in enumerate(w.shape)]
    g = g.reshape(rshape)
    w_wn = g * w * n
    return w_wn


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape, dim", [
    ((16, 8, 3, 3), 0), ((8, 3, 3, 16), 3), ((8, 16, 3, 3), 2),
    ((16, 8, 3), 0), ((8, 3, 3), 2), ((8, 16, 3), 1),
    ((16, 8), 0), ((8, 3), 1)
])
@pytest.mark.parametrize("eps", [1e-12])
def test_weight_normalization_forward_backward(eps, dim, shape, seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape)
    g = rng.randn(shape[dim])
    inputs = [w, g]
    func_args = [dim, eps]
    function_tester(rng, F.weight_normalization, ref_weight_normalization, inputs,
                    ctx=ctx, func_name=func_name, func_args=func_args, backward=[
                        True, True], disable_half_test=False,
                    atol_b=3e-3, atol_accum=3e-3)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape, dim", [
    ((2, 4, 16, 32), 0), ((32, 16, 4, 2), 3)])
@pytest.mark.parametrize("eps", [1e-12])
def test_large_weight_normalization(eps, dim, shape, seed, ctx, func_name):
    if not func_name.endswith('Cuda'):
        # This configuration is only to test the CUDA implementation branch
        # where reduction_size >= 1024, so we skip this test for
        # CPU and CuDNN and also do not need to run for both keepdims.
        pytest.skip('skip CUDA specific implementation test for other target')

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    w = rng.randn(*shape)
    g = rng.randn(shape[dim])
    inputs = [w, g]
    func_args = [dim, eps]
    function_tester(rng, F.weight_normalization, ref_weight_normalization, inputs,
                    ctx=ctx, func_name=func_name, func_args=func_args, backward=[
                        True, True],
                    atol_b=3e-3, atol_accum=3e-3)
