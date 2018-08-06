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
from nbla_test_utils import list_context


def ref_clip_by_value(x, min_, max_):
    min_idx = np.where(x < min_)
    x[min_idx] = min_[min_idx]
    max_idx = np.where(x > max_)
    x[max_idx] = max_[max_idx]
    return x


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shape", [(2, 8, 8, 8), (2, 3)])
def test_clip_by_value_forward(seed, shape):
    rng = np.random.RandomState(seed)
    x_data = rng.randn(*shape)
    min_data = rng.randn(*shape)
    max_data = rng.randn(*shape)
    x = nn.Variable.from_numpy_array(x_data)
    min_ = nn.Variable.from_numpy_array(min_data)
    max_ = nn.Variable.from_numpy_array(max_data)
    with nn.auto_forward(True):
        y = F.clip_by_value(x, min_, max_)
    y_ref = ref_clip_by_value(x_data, min_data, max_data)
    print(y.d)
    print(y_ref)
    assert np.allclose(y.d, y_ref)
