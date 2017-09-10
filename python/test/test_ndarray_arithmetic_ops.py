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


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**"])
@pytest.mark.parametrize("x_var, y_var", [(False, False), (False, True), (True, False)])
def test_ndarray_arithmetic_ops2(seed, op, x_var, y_var):
    rng = np.random.RandomState(seed)
    shape = [2, 3, 4]
    vx_data = rng.randn(*shape).astype(np.float32)
    vy_data = rng.randn(*shape).astype(np.float32)
    if op == "**":
        vx_data += - vx_data.min() + 1.0
    if x_var:
        vx = nn.Variable.from_numpy_array(vx_data)
    else:
        vx = nn.NdArray.from_numpy_array(vx_data)
    if y_var:
        vy = nn.Variable.from_numpy_array(vy_data)
    else:
        vy = nn.NdArray.from_numpy_array(vy_data)
    vz = eval("vx {0} vy".format(op))
    ref_z = eval("vx_data {0} vy_data".format(op))
    assert np.allclose(ref_z, vz.data)


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**"])
def test_ndarray_arithmetic_scalar_ops(seed, op):
    rng = np.random.RandomState(seed)
    vx = nn.NdArray.from_numpy_array(rng.randn(2, 3, 4).astype(np.float32))
    a = rng.randn()
    if op == "**":
        vx.data += - vx.data.min() + 1.0
    vz = eval("vx {0} a".format(op))
    ref_z = eval("vx.data {0} a".format(op))
    assert np.allclose(ref_z, vz.data)


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("op", ["+", "-", "*", "/", "**"])
def test_ndarray_arithmetic_scalar_rops(seed, op):
    rng = np.random.RandomState(seed)
    vx = nn.NdArray.from_numpy_array(rng.randn(2, 3, 4).astype(np.float32))
    a = rng.randn()
    if op == "**":
        a = np.abs(a)
    vz = eval("a {0} vx".format(op))
    ref_z = eval("a {0} vx.data".format(op))
    assert np.allclose(ref_z, vz.data)


@pytest.mark.parametrize("seed", [313, 314])
@pytest.mark.parametrize("op", ["+", "-"])
def test_ndarray_arithmetic_unary_ops(seed, op):
    rng = np.random.RandomState(seed)
    vx = nn.NdArray.from_numpy_array(rng.randn(2, 3, 4).astype(np.float32))
    vz = eval("{0} vx".format(op))
    ref_z = eval("{0} vx.data".format(op))
    assert np.allclose(ref_z, vz.data)
