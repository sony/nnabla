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


def test_manip():
    v = nn.Variable([2, 3, 4])
    assert v.shape == (2, 3, 4)
    with pytest.raises(Exception):
        v.reste_shape([1, 2])
    v.reset_shape([1, 2], force=True)
    assert v.shape == (1, 2)


@pytest.mark.parametrize("need_grad", [True, False])
def test_from_array(need_grad):
    data = np.random.randint(0, 10, size=(2, 3, 4))
    grad = np.random.randint(0, 10, size=(2, 3, 4))

    v1 = nn.Variable.from_numpy_array(data, need_grad=need_grad)
    assert np.all(v1.d == data)
    assert v1.d.dtype == data.dtype
    assert v1.need_grad == need_grad

    v2 = nn.Variable.from_numpy_array(data, grad, need_grad)
    assert np.all(v2.d == data)
    assert v2.d.dtype == data.dtype
    assert np.all(v2.g == grad)
    assert v2.g.dtype == grad.dtype
    assert v2.need_grad == need_grad


def test_data_grad_reference():
    v = nn.Variable([2, 3, 4])
    assert v.d.dtype == np.float32
    assert v.g.dtype == np.float32


def test_dtype_conversion():
    v = nn.Variable([2, 3, 4])
    a = v.data.cast(np.int)
    a[...] = 2
    assert (v.data.dtype == np.int)
    assert np.all(a == 2)
    b = v.data.cast(np.float32)
    assert b.dtype == np.float32
    assert b is not a
    assert np.all(b == 2)
    b[...] = np.random.randn(*b.shape) * 10
    c = v.data.cast(np.int32)
    assert np.all(c == b.astype(np.int32))


def test_data_grad():
    v = nn.Variable([2, 3, 4])
    v.d[...] = np.random.randn(*v.shape)
    assert v.d is not v.g
    assert not np.all(v.d == v.g)


def test_unlinked():
    v = nn.Variable([2, 3, 4], need_grad=True)
    grad = np.random.randn(*v.shape).astype(np.float32)
    v.g = grad
    v.d = np.random.randn(*v.shape)
    import nnabla.functions as F
    with nn.context_scope(nn.Context()), nn.auto_forward():
        v2 = F.identity(v)
        v2_u = v2.unlinked()
        v3 = F.identity(v2_u)
    v2_u.grad.zero()
    v2_g = v2_u.g.copy()
    v3.backward(clear_buffer=False)
    assert type(v2_u) == type(v2)
    assert np.all(v.g == grad)
    assert np.all(v2_u.g == v2.g)
    assert np.all(v2_u.g == v2_g + 1)


def test_rehape():
    v = nn.Variable([2, 3, 4], need_grad=True)
    grad = np.random.randn(*v.shape).astype(np.float32)
    v.g = grad
    v.d = np.random.randn(*v.shape)
    import nnabla.functions as F
    with nn.context_scope(nn.Context()), nn.auto_forward():
        v2 = F.identity(v)
        v2_s = v2.reshape((3, 4, 2))
        v3 = F.identity(v2_s)
    v3.backward(clear_buffer=False)
    assert np.all(v2_s.g.flat == v2.g.flat)
    assert np.all(v2_s.g == 1)
    v2.d = 1
    assert np.all(v2_s.d == 1)
    v2.g = 1.5
    assert np.all(v2_s.g == 1.5)


def test_persistent():
    x = nn.Variable([2, 3, 4], need_grad=True)
    x1 = x + 1
    x2 = x1 + 1
    x3 = x2 + 1
    y = x3 + 1
    x3.persistent = True
    x.data.zero()
    y.forward(clear_buffer=True)
    assert np.allclose(x3.d, 3)
    y.forward(clear_no_need_grad=True)
    y.backward(clear_buffer=True)
    assert np.allclose(x3.d, 3)
    assert np.allclose(x3.g, 1)
