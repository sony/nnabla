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


def ref_clip_by_norm(x, clip_norm, axis):
    x_norm = np.sqrt(np.sum(x ** 2.0, axis=axis, keepdims=True))
    x = clip_norm * x / x_norm
    return x


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("clip_norm", [0.5, ])
@pytest.mark.parametrize("shape, axis", [((2, 8, 8, 8), (2, 3)),
                                         ((2, 8, 8, 8), (1, )),
                                         ((2, 4), (1, )),
                                         ((2, 4, 4), None)])
def test_clip_by_norm_forward(seed, shape, clip_norm, axis):
    rng = np.random.RandomState(seed)
    x_data = rng.randn(*shape)
    x = nn.Variable.from_numpy_array(x_data)
    with nn.auto_forward(True):
        y = F.clip_by_norm(x, clip_norm, axis)
    y_ref = ref_clip_by_norm(x_data, clip_norm, axis=axis)
    assert np.allclose(y.d, y_ref)
