# Copyright 2021 Sony Group Corporation.
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


def ref_meshgrid(*x, ij_indexing):

    indexing = 'ij' if ij_indexing else 'xy'
    return np.meshgrid(*x, indexing=indexing)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("seed_num_arrays", [314])
@pytest.mark.parametrize("ij_indexing", [True, False])
@pytest.mark.parametrize("num_arrays", [2, 3, 4, 5])
@pytest.mark.parametrize("ctx, func_name", list_context('Meshgrid'))
def test_meshgrid(seed, seed_num_arrays, ij_indexing, num_arrays, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    rng_num_arrays = np.random.RandomState(seed_num_arrays)
    inputs = [rng.randn(rng_num_arrays.randint(1, 7), )
              for _ in range(num_arrays)]
    function_tester(rng, F.meshgrid, ref_meshgrid, inputs, func_kwargs=dict(ij_indexing=ij_indexing),
                    backward=[True]*num_arrays, ctx=ctx, func_name=func_name, disable_half_test=True, atol_b=0.03, atol_accum=1e-5)
