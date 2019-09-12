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
from nnabla.testing import assert_allclose


def test_imperative_i1_o1():
    import nnabla.functions as F
    x = nn.NdArray([2, 3, 4])
    x.fill(1)
    x1 = F.add_scalar(x, 1)
    assert_allclose(x1.data, 2)


def test_imperative_i2_o1():
    import nnabla.functions as F
    x0 = nn.NdArray([2, 3, 4])
    x1 = nn.NdArray([2, 1, 1])
    x0.fill(3)
    x1.fill(0.5)
    y = F.mul2(x0, x1)
    assert_allclose(y.data, 1.5)


def test_imperative_pf():
    import nnabla.parametric_functions as PF
    x = nn.NdArray([2, 3, 4, 5])
    y = PF.batch_normalization(x)
