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
import nnabla.random as R


@pytest.mark.parametrize("seed", [311])
def test_seed(seed):
    rand = F.rand(shape=(1, 2, 3))

    nn.seed(seed)

    # check Python random generator
    assert R.pseed == seed
    assert R.prng.rand() == np.random.RandomState(seed).rand()

    # check NNabla layer
    rand_values_before = []
    for i in range(10):
        rand.forward()
        rand_values_before.append(rand.d.copy())

    # reset seed
    nn.seed(seed)

    rand_values_after = []
    for i in range(10):
        rand.forward()
        rand_values_after.append(rand.d.copy())

    # check if global random generator is reset
    assert np.all(np.array(rand_values_before) == np.array(rand_values_after))
