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
from nnabla.testing import assert_allclose


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**"])
def test_variable_arithmetic_ops2(seed, op):
    rng = np.random.RandomState(seed)
    shape = [2, 3, 4]
    vx = nn.Variable.from_numpy_array(rng.randn(*shape).astype(np.float32))
    vy = nn.Variable.from_numpy_array(rng.randn(*shape).astype(np.float32))
    if op == "**":
        vx.d += - vx.d.min() + 1.0
    with nn.auto_forward():
        vz = eval("vx {0} vy".format(op))
        ref_z = eval("vx.d {0} vy.d".format(op))
        assert_allclose(ref_z, vz.d)

    # Inplace test
    with nn.auto_forward():
        # Make function reference count of `vx` to 1.
        vx = nn.functions.identity(vx)
        vx_bak = vx
        if op == "+":
            vx += vy
        elif op == "-":
            vx -= vy
        elif op == "*":
            vx *= vy
        elif op == "/":
            vx /= vy
        elif op == "**":
            vx **= vy
        assert_allclose(vx.d, vz.d)
        assert vx is not vx_bak


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**"])
def test_variable_arithmetic_scalar_ops(seed, op):
    rng = np.random.RandomState(seed)
    vx = nn.Variable.from_numpy_array(rng.randn(2, 3, 4).astype(np.float32))
    a = rng.randn()
    if op == "**":
        vx.d += - vx.d.min() + 1.0
    with nn.auto_forward():
        vz = eval("vx {0} a".format(op))
        ref_z = eval("vx.d {0} a".format(op))
        assert_allclose(ref_z, vz.d)

    # Inplace test
    with nn.auto_forward():
        # Make function reference count of `vx` to 1.
        vx = nn.functions.identity(vx)
        vx_bak = vx
        if op == "+":
            vx += a
        elif op == "-":
            vx -= a
        elif op == "*":
            vx *= a
        elif op == "/":
            vx /= a
        elif op == "**":
            vx **= a
        assert_allclose(vx.d, vz.d)
        assert vx is not vx_bak


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**"])
def test_variable_arithmetic_scalar_rops(seed, op):
    rng = np.random.RandomState(seed)
    vx = nn.Variable.from_numpy_array(rng.randn(2, 3, 4).astype(np.float32))
    a = rng.randn()
    if op == "**":
        a = np.abs(a)
    with nn.auto_forward():
        vz = eval("a {0} vx".format(op))
        ref_z = eval("a {0} vx.d".format(op))
        assert_allclose(ref_z, vz.d)


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("op", ["+", "-"])
def test_variable_arithmetic_unary_ops(seed, op):
    rng = np.random.RandomState(seed)
    vx = nn.Variable.from_numpy_array(rng.randn(2, 3, 4).astype(np.float32))
    with nn.auto_forward():
        vz = eval("{0} vx".format(op))
        ref_z = eval("{0} vx.d".format(op))
        assert_allclose(ref_z, vz.d)
