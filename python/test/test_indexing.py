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


class Key2Key:
    def __getitem__(self, key):
        return key


def allclose(a, b):
    assert isinstance(a, (nn.Variable, nn.NdArray))
    assert isinstance(b, (np.ndarray, int, float))
    return np.allclose(a.data if isinstance(a, nn.NdArray) else a.d, b)


@pytest.mark.parametrize("key", [
    "10:19:-1",
    "10::-1",
    "10:20:-1",
    "1",
    ":",
    "3:8",
    "3:16:3",
    "...",
    "np.newaxis",
    "[1,4,7,18]",
    "(1,4,7,18),",
    "np.array([4,7])",
    "[1,4,7,18], np.newaxis",
    "np.newaxis, [1,4,7,18], np.newaxis",
    "x_np > 0.5",
    "x_np > 0.5, np.newaxis",
    "np.newaxis, x_np > 0.5, np.newaxis",
])
@pytest.mark.parametrize("nn_type", [nn.NdArray, nn.Variable])
def test_1d_array_indexing(key, nn_type):
    x_np = np.random.rand(20)
    x_nn = nn_type.from_numpy_array(x_np)

    with nn.auto_forward(True):
        y_nn = x_nn.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        y_np = x_np.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        assert allclose(y_nn, y_np)

    with nn.auto_forward(True):
        data = np.random.rand(*eval("x_np[{key}].shape".format(key=key)))
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        assert allclose(x_nn, x_np)

    with nn.auto_forward(True):
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        assert allclose(x_nn, x_np)


@pytest.mark.parametrize("key", [
    ":, :, 4:36, 4:36",
    ":, :2",
    "2, :",
    ":, :, 1:-1, 1:-1",
    ":, :, 1:-2, 1:-2",
    ":, :, 1:-3, 1:-3",
    ":, 0, :, :",
    "3, ...",
    "..., 3",
    "3, 0, :, :",
    "...",
    "[1,2,3,4]",
    "[1,2,3,4], [2,1,4,3]",
    "[1,2], [3,4], [2,1], [4,3]",
    "[1,2,3,4], None",
    "None, [1,2,3,4]",
    "[1,2,3,4], [2,1,4,3], None",
    "None, [1,2,3,4], [2,1,4,3]",
    "[1,2], None, [3,4], None, [2,1]",
    "[1,2], None, [3,4], [2,1], None, [4,3]",
    "x_np > 0.5",
    "x_np > 0.5, np.newaxis",
    "np.newaxis, x_np > 0.5, np.newaxis",
])
@pytest.mark.parametrize("nn_type", [nn.NdArray, nn.Variable])
def test_4d_array_indexing(key, nn_type):
    b, c, h, w = 8, 16, 40, 40
    x_np = np.random.rand(b, c, h, w)
    x_nn = nn_type.from_numpy_array(x_np)

    with nn.auto_forward(True):
        y_nn = x_nn.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        y_np = x_np.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        assert allclose(y_nn, y_np)

    with nn.auto_forward(True):
        data = np.random.rand(*eval("x_np[{key}].shape".format(key=key)))
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        assert allclose(x_nn, x_np)

    with nn.auto_forward(True):
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        assert allclose(x_nn, x_np)


@pytest.mark.parametrize("key", [
    ":, :, :, ::-1",
    ":, :, ::-1, ::-1",
])
@pytest.mark.parametrize("nn_type", [nn.NdArray, nn.Variable])
def test_flipping(key, nn_type):
    b, c, h, w = 8, 8, 16, 16
    x_np = np.random.rand(b, c, h, w)
    x_nn = nn_type.from_numpy_array(x_np)

    with nn.auto_forward(True):
        y_nn = x_nn.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        y_np = x_np.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        assert allclose(y_nn, y_np)

    with nn.auto_forward(True):
        data = np.random.rand(*eval("x_np[{key}].shape".format(key=key)))
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        assert allclose(x_nn, x_np)

    with nn.auto_forward(True):
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        assert allclose(x_nn, x_np)


@pytest.mark.parametrize("key", [
    "1",
    ":, ..., :",
    ":, 3, 0, ..., :",
    ":, 3, 0, ..., 2:5:2, 3",
    ":, 3, 0, ..., 3, :",
    ":, 3, ..., :",
    "np.newaxis",
    "0:5:3, 3, np.newaxis, 0, ..., 3, :",
    ":, 3, ..., np.newaxis, 3, :",
    ":, 3, ..., 3, np.newaxis, :",
    ":, 3, ..., 3, :, np.newaxis",
    "np.newaxis, :, 3, ..., 3, :",
])
@pytest.mark.parametrize("nn_type", [nn.NdArray, nn.Variable])
def test_6d_array_indexing(key, nn_type):
    b, c, s0, s1, s2, s3 = 8, 16, 1, 7, 6, 4
    x_np = np.random.rand(b, c, s0, s1, s2, s3)
    x_nn = nn.NdArray.from_numpy_array(x_np)

    with nn.auto_forward(True):
        y_nn = x_nn.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        y_np = x_np.__getitem__(eval("Key2Key()[{key}]".format(key=key)))
        assert allclose(y_nn, y_np)

    with nn.auto_forward(True):
        data = np.random.rand(*eval("x_np[{key}].shape".format(key=key)))
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), data)
        assert allclose(x_nn, x_np)

    with nn.auto_forward(True):
        # The next two lines must not be swapped because a boolean key like
        # "x_np > 0.5" must first be used for x_nn before it modifies x_np.
        x_nn.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        x_np.__setitem__(eval("Key2Key()[{key}]".format(key=key)), 69)
        assert allclose(x_nn, x_np)


def test_update_index_variable():
    x_np = np.random.rand(20)
    i_np = np.random.choice(np.arange(20), 10)
    x_nn = nn.Variable.from_numpy_array(x_np)
    i_nn = nn.Variable.from_numpy_array(i_np)
    y_nn = x_nn[i_nn]
    y_nn.forward()
    assert_allclose(y_nn.d, x_np[i_np])
    i_np = np.random.choice(np.arange(20), 10)
    i_nn.d = i_np
    y_nn.forward()
    assert_allclose(y_nn.d, x_np[i_np])


def test_nnabla_boolean_array_as_index():
    x_np = np.random.rand(20)
    i_np = x_np > 0.5
    x_nn = nn.Variable.from_numpy_array(x_np)
    i_nn = nn.Variable.from_numpy_array(i_np)
    with nn.auto_forward(True):
        assert_allclose(x_nn[i_nn].d, x_np[i_np])
        assert_allclose(x_nn.data[i_nn].data, x_np[i_np])
        x_np[i_np] = -1
        x_nn[i_nn] = -1
        assert_allclose(x_nn.d, x_np)


@pytest.mark.parametrize("indices", [
    ([[2, 2], [3, 3], [4, 4]], [[3, 4], [3, 4], [3, 4]]),
    (nn.Variable.from_numpy_array([[2, 2], [3, 3], [4, 4]]),
     nn.Variable.from_numpy_array([[3, 4], [3, 4], [3, 4]])),
    (slice(2, 5), slice(3, 5)),
])
def test_graph_connection_with_setitem(indices):
    import nnabla.functions as F
    x = np.arange(8 * 7).reshape((8, 7))
    x = nn.Variable.from_numpy_array(x, need_grad=True)
    u = np.arange(-1, -7, -1).reshape(3, 2)
    u = nn.Variable.from_numpy_array(u, need_grad=True)
    y = F.mul_scalar(x, 1)
    y[indices] = u
    z = F.add_scalar(y, 0)
    z.forward()
    # '+' signs only to persist visual alignment through autopep8
    assert_allclose(z.d, np.array([[+0, +1, +2, +3, +4, +5, +6],
                                   [+7, +8, +9, 10, 11, 12, 13],
                                   [14, 15, 16, -1, -2, 19, 20],
                                   [21, 22, 23, -3, -4, 26, 27],
                                   [28, 29, 30, -5, -6, 33, 34],
                                   [35, 36, 37, 38, 39, 40, 41],
                                   [42, 43, 44, 45, 46, 47, 48],
                                   [49, 50, 51, 52, 53, 54, 55]]))
    x.grad.zero()
    u.grad.zero()
    z.backward(np.arange(1, 1 + 8 * 7).reshape(8, 7))
    assert_allclose(x.g, np.array([[+1, +2, +3, +4, +5, +6, +7],
                                   [+8, +9, 10, 11, 12, 13, 14],
                                   [15, 16, 17, +0, +0, 20, 21],
                                   [22, 23, 24, +0, +0, 27, 28],
                                   [29, 30, 31, +0, +0, 34, 35],
                                   [36, 37, 38, 39, 40, 41, 42],
                                   [43, 44, 45, 46, 47, 48, 49],
                                   [50, 51, 52, 53, 54, 55, 56]]))
    assert_allclose(u.g, np.array([[18, 19],
                                   [25, 26],
                                   [32, 33]]))
