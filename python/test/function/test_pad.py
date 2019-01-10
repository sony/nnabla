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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('Pad')


def ref_pad_constant(x, pad_width, mode, constant_value):
    pad_width = list(zip(pad_width[::2], pad_width[1::2]))
    pad_width = (x.ndim - len(pad_width)) * [(0, 0)] + pad_width
    return np.pad(x, pad_width, mode, constant_values=constant_value)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, pad_width", [
    ((4,), (2, 9)),
    ((5,), (4, 3)),
    ((4, 5), (5, 4)),
    ((3, 6), (1, 2, 3, 4)),
    ((3, 5, 7), (2, 3)),
    ((5, 4, 6), (2, 3, 1, 4)),
    ((2, 3, 5), (1, 2, 3, 4, 5, 6)),
    ((1, 2, 3, 4), (2, 3, 4, 5)),
    ((2, 2, 3, 3), (2, 2, 2, 2, 2, 2, 2, 2)),
])
@pytest.mark.parametrize("constant_value", [0.11, -1.1])
def test_pad_constant_forward_backward(seed, ctx, func_name, inshape,
                                       pad_width, constant_value):
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [pad_width, "constant", constant_value]
    function_tester(rng, F.pad, ref_pad_constant, inputs, ctx=ctx, dstep=1e-1,
                    func_name=func_name, func_args=func_args)


def ref_pad_reflect(x, pad_width, mode):
    pad_width = list(zip(pad_width[::2], pad_width[1::2]))
    pad_width = (x.ndim - len(pad_width)) * [(0, 0)] + pad_width
    return np.pad(x, pad_width, mode)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape, pad_width", [
    ((4,), (2, 9)),
    ((5,), (4, 3)),
    ((4, 5), (5, 4)),
    ((3, 6), (1, 2, 3, 4)),
    ((3, 5, 7), (2, 3)),
    ((5, 4, 6), (2, 3, 1, 4)),
    ((2, 3, 5), (1, 2, 3, 4, 5, 6)),
    ((1, 2, 3, 4), (2, 3, 4, 5)),
    ((2, 2, 3, 3), (2, 2, 2, 2, 2, 2, 2, 2)),
])
def test_pad_reflect_forward_backward(seed, ctx, func_name, inshape,
                                      pad_width):
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [pad_width, "reflect"]
    function_tester(rng, F.pad, ref_pad_reflect, inputs, ctx=ctx, dstep=1e-1,
                    func_name=func_name, func_args=func_args)
