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


def test_1d_array_indexing():
    # 1d-tensor
    d = 16
    x_data = np.random.rand(d)
    x = nn.NdArray.from_numpy_array(x_data)

    with nn.auto_forward(True):
        x_data_key = x_data[1]
        x_key = x[1]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:]
        x_key = x[:]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[3:8]
        x_key = x[3:8]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[3:16:3]
        x_key = x[3:16:3]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[...]
        x_key = x[...]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[np.newaxis]
        x_key = x[np.newaxis]
        assert np.allclose(x_key.data, x_data_key)


def test_4d_array_indexing():
    # 4d-tensor
    b, c, h, w = 8, 16, 40, 40
    x_data = np.random.rand(b, c, h, w)
    x = nn.NdArray.from_numpy_array(x_data)

    with nn.auto_forward(True):
        x_data_key = x_data[:, :, 4:36, 4:36]
        x_key = x[:, :, 4:36, 4:36]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 0, :, :]
        x_key = x[:, 0, :, :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[3, ...]
        x_key = x[3, ...]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[3, 0, :, :]
        x_key = x[3, 0, :, :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[3, ...]
        x_key = x[3, ...]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[...]
        x_key = x[...]
        assert np.allclose(x_key.data, x_data_key)


def test_complex_nd_array_indexing():
    # Pseudo-Complex
    b, h, w, cr = 8, 16, 16, 2
    x_data = np.random.rand(b, h, w, cr)
    x = nn.NdArray.from_numpy_array(x_data)

    with nn.auto_forward(True):
        x_data_key = x_data[..., 0]
        x_key = x[..., 0]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[..., 1]
        x_key = x[..., 1]
        assert np.allclose(x_key.data, x_data_key)


def test_6d_array_indexing():
    # Hard
    b, c, s0, s1, s2, s3 = 8, 16, 1, 7, 6, 4
    x_data = np.random.rand(b, c, s0, s1, s2, s3)
    x = nn.NdArray.from_numpy_array(x_data)

    with nn.auto_forward(True):
        x_data_key = x_data[1]
        x_key = x[1]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, ..., :]
        x_key = x[:, ..., :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 3, 0, ..., :]
        x_key = x[:, 3, 0, ..., :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 3, 0, ..., 2:5:2, 3]
        x_key = x[:, 3, 0, ..., 2:5:2, 3]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 3, 0, ..., 3, :]
        x_key = x[:, 3, 0, ..., 3, :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 3, ..., :]
        x_key = x[:, 3, ..., :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[np.newaxis]
        x_key = x[np.newaxis]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[0:5:3, 3, np.newaxis, 0, ..., 3, :]
        x_key = x[0:5:3, 3, np.newaxis, 0, ..., 3, :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 3, ..., np.newaxis, 3, :]
        x_key = x[:, 3, ..., np.newaxis, 3, :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 3, ..., 3, np.newaxis, :]
        x_key = x[:, 3, ..., 3, np.newaxis, :]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[:, 3, ..., 3, :, np.newaxis]
        x_key = x[:, 3, ..., 3, :, np.newaxis]
        assert np.allclose(x_key.data, x_data_key)

        x_data_key = x_data[np.newaxis, :, 3, ..., 3, :]
        x_key = x[np.newaxis, :, 3, ..., 3, :]
        assert np.allclose(x_key.data, x_data_key)
