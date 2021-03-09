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
def test_pad_constant_double_backward(seed, ctx, func_name, inshape,
                                      pad_width, constant_value):
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.pad import PadDataGrad
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    func_args = [pad_width, "constant", constant_value]
    # 2nd-order
    backward_function_tester(rng, F.pad, inputs, ctx=ctx, func_args=func_args)

    # 3rd-order
    # constant value is always zero after 1st-order derivative
    func_args = [pad_width, "constant", 0]
    df, y = grad_function_forward_function_output(PadDataGrad,
                                                  F.pad,
                                                  ctx, inputs,
                                                  *func_args)
    df.xshape = inputs[0].shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             ginputs, func_args=[],
                             ctx=ctx, atol_f=1e-6, atol_accum=5e-2,
                             non_accum_check=True)


# @pytest.mark.parametrize("seed", [313])
# @pytest.mark.parametrize("ctx, func_name", ctxs)
# @pytest.mark.parametrize("inshape, pad_width", [
##     ((4,), (2, 9)),
##     ((5,), (4, 3)),
##     ((4, 5), (5, 4)),
##     ((3, 6), (1, 2, 3, 4)),
##     ((3, 5, 7), (2, 3)),
##     ((5, 4, 6), (2, 3, 1, 4)),
##     ((2, 3, 5), (1, 2, 3, 4, 5, 6)),
##     ((1, 2, 3, 4), (2, 3, 4, 5)),
##     ((2, 2, 3, 3), (2, 2, 2, 2, 2, 2, 2, 2)),
# ])
# def test_pad_reflect_double_backward(seed, ctx, func_name, inshape,
# pad_width):
##     from nbla_test_utils import backward_function_tester
##     rng = np.random.RandomState(seed)
##     inputs = [rng.randn(*inshape).astype(np.float32)]
##     func_args = [pad_width, "reflect"]
# backward_function_tester(rng, F.pad, ref_pad_reflect, inputs, ctx=ctx,
# dstep=1e-3, atol_b=5e-2, atol_accum=5e-2,
# func_name=func_name, func_args=func_args)
