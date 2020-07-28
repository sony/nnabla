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

from nbla_test_utils import (
    function_tester,
    list_ctx_and_func_name)


def ref_broadcast(x, shape):
    return x * np.ones(shape, dtype=x.dtype)


def get_combination(n):
    if n == 0:
        return [(n, np.array([], dtype=np.bool))]
    all_comb = np.vstack(map(lambda x: x.flatten(), np.meshgrid(
        *[[0, 1] for _ in range(n)]))).T.astype(np.bool)
    return [(n, comb) for comb in all_comb]


def get_combinations(*N):
    ret = []
    for n in N:
        ret.extend(get_combination(n))
    return ret


@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name", list_ctx_and_func_name(['broadcast']))
@pytest.mark.parametrize("ndim, broadcast_dim", get_combinations(*range(0, 6)))
def test_broadcast_forward_backward(ndim, broadcast_dim, seed, fname, ctx, func_name):
    func = getattr(F, fname)
    ref_func = eval('ref_' + fname)
    rng = np.random.RandomState(seed)
    shape = rng.randint(2, 5, size=(ndim,))
    inshape = shape.copy()
    inshape[broadcast_dim] = 1
    if ndim == 0:
        # Performing 0-dim array test too.
        inputs = [np.array(rng.randn()).astype("float32")]
        function_tester(rng, func, ref_func, inputs, [shape],
                        ctx=ctx, backward=[True], func_name=func_name,
                        atol_b=4e-3)

    inputs = [np.array(rng.randn(*inshape)).astype("float32")]
    function_tester(rng, func, ref_func, inputs, [shape],
                    ctx=ctx, backward=[True], func_name=func_name,
                    atol_b=6e-3)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("fname, ctx, func_name", list_ctx_and_func_name(['broadcast']))
@pytest.mark.parametrize("ndim, broadcast_dim", get_combinations(*range(0, 6)))
def test_broadcast_double_backward(ndim, broadcast_dim, seed, fname, ctx, func_name):
    from nbla_test_utils import cap_ignore_region, backward_function_tester

    rng = np.random.RandomState(seed)
    shape = rng.randint(2, 5, size=(ndim,))
    inshape = shape.copy()
    inshape[broadcast_dim] = 1
    if ndim == 0:
        # Performing 0-dim array test too.
        inputs = [np.array(rng.randn()).astype("float32")]
        backward_function_tester(rng, F.broadcast,
                                 inputs=inputs,
                                 func_args=[shape], func_kwargs={},
                                 ctx=ctx)

    inputs = [np.array(rng.randn(*inshape)).astype("float32")]
    backward_function_tester(rng, F.broadcast, inputs,
                             func_args=[shape], func_kwargs={},
                             dstep=1e-3,
                             ctx=ctx)
