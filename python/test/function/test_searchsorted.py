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


def ref_searchsorted(sorted_sequence, values, right):

    assert sorted_sequence.ndim == values.ndim, 'ndim of sorted_sequence and values array must be same'

    for i in range(values.ndim-1):
        assert sorted_sequence.shape[i] == values.shape[
            i], f'inner dimensions of sorted sequence and values must match. {sorted_sequence.shape[i]}!={values.shape[i]}'

    side = 'right' if right else 'left'
    values_shape = values.shape

    sorted_sequence = sorted_sequence.reshape(-1, sorted_sequence.shape[-1])
    values = values.reshape(-1, values.shape[-1])

    out_list = []
    for i in range(sorted_sequence.shape[0]):
        out_list.append(np.searchsorted(
            sorted_sequence[i], values[i], side=side))
    out = np.concatenate([out_[None, :] for out_ in out_list], axis=0)
    return out.reshape(values_shape)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("low", [-1, 0])
@pytest.mark.parametrize("high", [1, 10])
@pytest.mark.parametrize("num_values", [1, 5, 10, 20])
@pytest.mark.parametrize("ctx, func_name", list_context('SearchSorted'))
def test_searchsorted_forward(seed, right, low, high, num_values, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [np.sort((rng.randint(low=low, high=high, size=(3, 5, 7, 10)))), rng.randint(
        low=-10, high=20, size=(3, 5, 7, num_values))]
    function_tester(rng, F.searchsorted, ref_searchsorted, inputs, func_args=[right],
                    backward=[False, False], ctx=ctx, func_name=func_name, disable_half_test=True)
