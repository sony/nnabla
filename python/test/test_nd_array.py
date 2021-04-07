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

import numpy as np

import nnabla as nn

import pytest


def test_nd_array():
    shape = [2, 3, 4]
    a = nn.NdArray(shape)
    npa = np.arange(a.size).reshape(a.shape).astype(np.int32)
    a.data = npa
    b = nn.NdArray.from_numpy_array(npa)
    b.dtype == np.int32
    assert np.all(a.data == npa)
    assert np.all(a.data == b.data)
    assert a.shape == npa.shape
    assert b.size == np.prod(shape)
    a.cast(np.int32)
    assert a.data.dtype == np.int32
    b.zero()
    assert np.all(b.data == 0)
    a.fill(3)
    assert np.all(a.data == 3)
    b.copy_from(a)
    assert np.all(a.data == b.data)


def test_copy_from():
    shape = [2, 3, 4]
    src = nn.NdArray(shape)
    dst = nn.NdArray(shape)
    src.data = 0
    src.cast(dtype=np.uint8)
    dst.copy_from(src, use_current_context=False)
    assert dst.dtype == np.uint8

    from nnabla.ext_utils import get_extension_context
    with nn.context_scope(get_extension_context('cpu', dtype='float')):
        dst.copy_from(src, use_current_context=True)
    assert dst.dtype == np.float32


@pytest.mark.parametrize("value", [
    1,
    1.3,
    np.array(np.zeros((2, 3))),
    np.arange(6).reshape(2, 3)])
def test_nd_array_data(value):
    shape = (2, 3)

    # Use default dtype (float32) in getter
    a = nn.NdArray(shape)
    with pytest.raises(Exception):
        _ = a.dtype
    _ = a.data
    assert a.dtype == np.float32

    # Use value dtype in setter
    a = nn.NdArray(shape)
    a.data = value
    if not np.isscalar(value) or \
       (np.dtype(type(value)).kind != 'f' and value > (1 << 53)):
        assert a.dtype == np.asarray(value).dtype
        assert a.data.dtype == np.asarray(value).dtype
    else:
        assert a.data.dtype == np.float32


def test_clear_called():
    a = nn.NdArray(1)
    assert a.clear_called == False
    a.fill(3)
    assert a.clear_called == False
    a.clear()
    assert a.clear_called == True

    a.fill(3)
    assert a.clear_called == False
    a.clear()
    assert a.clear_called == True
    a.zero()
    assert a.clear_called == False
    a.clear()
    assert a.clear_called == True

    a.data[0] = -1
    assert a.clear_called == False
