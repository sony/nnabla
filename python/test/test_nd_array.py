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
