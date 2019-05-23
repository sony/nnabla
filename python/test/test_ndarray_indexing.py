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
def test_1d_array_indexing(key):
    x_np = np.random.rand(20)
    x_nn = nn.NdArray.from_numpy_array(x_np)
    with nn.auto_forward(True):
        assert np.allclose(eval("x_nn[{key}].data".format(key=key)),
                           eval("x_np[{key}]".format(key=key)))


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
    "x_np > 0.5",
    "x_np > 0.5, np.newaxis",
    "np.newaxis, x_np > 0.5, np.newaxis",
])
def test_4d_array_indexing(key):
    b, c, h, w = 8, 16, 40, 40
    x_np = np.random.rand(b, c, h, w)
    x_nn = nn.NdArray.from_numpy_array(x_np)
    with nn.auto_forward(True):
        assert np.allclose(eval("x_nn[{key}].data".format(key=key)),
                           eval("x_np[{key}]".format(key=key)))


@pytest.mark.parametrize("key", [
    ":, :, :, ::-1",
    ":, :, ::-1, ::-1",
])
def test_flipping(key):
    b, c, h, w = 8, 8, 16, 16
    x_np = np.random.rand(b, c, h, w)
    x_nn = nn.NdArray.from_numpy_array(x_np)
    with nn.auto_forward(True):
        assert np.allclose(eval("x_nn[{key}].data".format(key=key)),
                           eval("x_np[{key}]".format(key=key)))


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
def test_6d_array_indexing(key):
    # Hard
    b, c, s0, s1, s2, s3 = 8, 16, 1, 7, 6, 4
    x_np = np.random.rand(b, c, s0, s1, s2, s3)
    x_nn = nn.NdArray.from_numpy_array(x_np)
    with nn.auto_forward(True):
        assert np.allclose(eval("x_nn[{key}].data".format(key=key)),
                           eval("x_np[{key}]".format(key=key)))


def test_update_index_variable():
    x_np = np.random.rand(20)
    i_np = np.random.choice(np.arange(20), 10)
    x_nn = nn.Variable.from_numpy_array(x_np)
    i_nn = nn.Variable.from_numpy_array(i_np)
    y_nn = x_nn[i_nn]
    y_nn.forward()
    assert np.allclose(y_nn.d, x_np[i_np])
    i_np = np.random.choice(np.arange(20), 10)
    i_nn.d = i_np
    y_nn.forward()
    assert np.allclose(y_nn.d, x_np[i_np])
