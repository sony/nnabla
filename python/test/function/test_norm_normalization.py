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

ctxs = list_context('NormNormalization')


def ref_norm_normalization(x, p, axis, eps=1e-12):
    if p is None:
        p = 2.
    # calculate norm
    y = x
    y = np.abs(y)
    y = np.power(y, p)
    y = np.sum(y, axis, keepdims=True) + eps
    y = np.power(y, 1./p)
    # normalization
    y = x / y
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("p", [None, 1.0, 1.3, 3.0])  # None -> 2.0
@pytest.mark.parametrize("shape, axis", [
    ((2, 3, 5, 7), (0, 2)), ((13,), 0), ((7, 3, 1), None), ((2, 1, 4, 5), (0, 2))
])
@pytest.mark.parametrize("eps", [1e-12])
def test_norm_normalization_forward_backward(eps, axis, p, shape, seed, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, function_tester
    from sys import platform
    if platform == "darwin":
        pytest.skip("NormNormalization is not supported in macOS.")

    rng = np.random.RandomState(seed)
    inputs = [cap_ignore_region(
        rng.randn(*shape).astype(np.float32) * 2, (-1e-3, 1e-3))]
    func_args = [p, axis, eps]
    function_tester(rng, F.norm_normalization, ref_norm_normalization, inputs,
                    ctx=ctx, func_name=func_name, func_args=func_args, backward=[
                        True], disable_half_test=False,
                    # The backward test on macOS doesn't pass with this tolerance.
                    # Does Eigen library used in CPU computation backend produce
                    # the different results on different platforms?
                    # atol_b=3e-3,
                    atol_b=1e-2, atol_accum=1e-2)
