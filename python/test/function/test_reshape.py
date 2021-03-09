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
from nbla_test_utils import list_context

ctxs = list_context('Reshape')


def ref_reshape(x, shape, inplace):
    return x.reshape(shape).copy()


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, outshape", [((1, 1, 6), (1, 2, 3)), ((2, 3), (1, 6)), ((2, 4), (-1, 2, 2))])
def test_reshape_forward_backward(seed, inshape, outshape, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(*inshape).astype(np.float32)]
    inplace = False
    function_tester(rng, F.reshape, ref_reshape, inputs, func_args=[
                    outshape, inplace], ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, outshape", [((1, 1, 6), (1, 2, 3)), ((2, 3), (1, 6)), ((2, 4), (-1, 2, 2))])
def test_reshpae_inplace(seed, ctx, func_name, inshape, outshape):
    from nbla_test_utils import inplace_function_test_helper
    rng = np.random.RandomState(seed)
    inputs = [nn.Variable.from_numpy_array(
        rng.randn(*inshape).astype(np.float32))]
    inplace_function_test_helper(
        inputs, F.reshape, func_args=[
            outshape], ctx=ctx, func_name=func_name, rng=np.random.RandomState(seed))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, outshape", [((1, 1, 6), (1, 2, 3)), ((2, 3), (1, 6)), ((2, 4), (-1, 2, 2))])
@pytest.mark.parametrize("inplace", [False, True])
def test_reshape_double_backward(seed, ctx, func_name, inshape, outshape, inplace):
    from nbla_test_utils import cap_ignore_region, backward_function_tester
    rng = np.random.RandomState(seed)
    # Input
    inputs = [rng.randn(*inshape).astype(np.float32)]
    # TODO: backward_ref
    backward_function_tester(rng, F.reshape,
                             inputs=inputs,
                             func_args=[outshape, inplace], func_kwargs={},
                             atol_accum=5e-3,
                             dstep=1e-3,
                             ctx=ctx)
