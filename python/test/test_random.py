# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
from nnabla.testing import assert_allclose


@pytest.mark.parametrize("seed", [313, 314])
def test_set_seed(seed):
    # reference random generator
    rng = np.random.RandomState(seed)

    # set nnabla seed
    nn.set_seed(seed)

    # sample random values to check the same seed
    assert_allclose(rng.rand(3, 2, 1), nn.random.prng.rand(3, 2, 1))
    assert_allclose(rng.rand(3, 2, 1), nn.random.prng.rand(3, 2, 1))
    assert_allclose(rng.rand(3, 2, 1), nn.random.prng.rand(3, 2, 1))
