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

import pytest
import numpy as np
import nnabla.functions as F
from nbla_test_utils import list_context, function_tester, half_test, force_tuple
from nnabla.testing import assert_allclose

import nnabla as nn
import nnabla.ext_utils as ext_utils

ctxs = list_context("Linspace")

test_data = [
    (0, 10, 1),
    (5, 10, 5),
    (10, 0, 2),
    (10, 100, 17),
    (-50, 50, 23),
    (0, 10, 100000),
    (0, 100000, 10),
    (0, 10, 0),
    (0, 0, 3),
]


# test for float32 (single-precision)
# use numpy.linspace(dtype=float32) as a reference function


def ref_linspace(start, stop, num):
    return np.linspace(start, stop, num).astype(np.float32)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize(
    "start, stop, num",
    test_data,
)
def test_linspace_forward(start, stop, num, ctx, func_name):
    function_tester(
        None,
        F.linspace,
        ref_linspace,
        inputs=[],
        ctx=ctx,
        func_args=[start, stop, num],
        func_name=func_name,
        backward=[],
        disable_half_test=True,
    )


# test for float16 (half-precision)
# use numpy.linspace(dtype=float16) as a reference function


def ref_linspace_half(start, stop, num):
    return np.linspace(start, stop, num).astype(np.float16)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize(
    "start, stop, num",
    test_data,
)
def test_linspace_forward_half(start, stop, num, ctx, func_name):
    ext, dtype = ctx.backend[0].split(":")
    assert dtype == "float"
    ctx_h = ext_utils.get_extension_context(ext, type_config="half")
    ctx_h.device_id = ctx.device_id
    with nn.context_scope(ctx_h):
        o_h = force_tuple(F.linspace(start, stop, num))

    o_h[0].parent.forward([], o_h)
    y_h = [o.d.copy() for o in o_h]
    y_np = force_tuple(ref_linspace_half(start, stop, num))

    # numpy.linspace function is running with dtype=float16,
    # therefore the output results will usually match.
    # If it does not, check the algorithm and whether it is acceptable.
    for y, l in zip(y_h, y_np):
        assert y.all() == l.all()


# test for runtime errors
# if `num` value is a real number, raise TypeError


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize(
    "start, stop, num",
    [
        (0, 10, 2.5),
    ],
)
def test_linspace_forward_type_errors(start, stop, num, ctx, func_name):
    with pytest.raises(TypeError):
        F.linspace(start, stop, num)


# test for runtime errors
# if `num` value is negative, raise ValueError


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize(
    "start, stop, num",
    [
        (0, 10, -1),
    ],
)
def test_linspace_forward_value_errors(start, stop, num, ctx, func_name):
    with pytest.raises(ValueError):
        F.linspace(start, stop, num)
