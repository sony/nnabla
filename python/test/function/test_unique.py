# Copyright 2018,2019,2020,2021 Sony Corporation.
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
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Unique')


def ref_unique(x, flatten, axis, sorted, with_index, with_inverse, with_counts):
    if flatten:
        axis = None
    y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis)

    if not sorted:
        argsort_indices = np.argsort(indices)
        inverse_arg = {
            arg_val: arg_index for arg_index, arg_val in enumerate(argsort_indices)
        }
        indices = indices[argsort_indices]
        counts = counts[argsort_indices]
        y = np.take(x, indices, axis=axis)
        inverse_indices = np.asarray([inverse_arg[i] for i in inverse_indices])

    outputs = (y,)
    indices = indices.astype(np.int64)
    inverse_indices = inverse_indices.astype(np.int64)
    counts = counts.astype(np.int64)
    if with_index:
        outputs += (indices,)
    if with_inverse:
        outputs += (inverse_indices,)
    if with_counts:
        outputs += (counts,)
    return outputs


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("x_shape, axis", [
    ((1,), 0),
    ((2, 3, 4), 1),
    ((2, 3, 4, 5), 3),
])
@pytest.mark.parametrize("flatten", [True, False])
@pytest.mark.parametrize("sorted", [True, False])
@pytest.mark.parametrize("with_index, with_inverse, with_counts", [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (False, False, False),
])
@pytest.mark.parametrize("seed", [313])
def test_unique_forward(seed, x_shape, flatten, axis, sorted, with_index, with_inverse, with_counts, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randint(-10, 10, x_shape).astype(np.float32)]
    func_args = [flatten, axis, sorted, with_index, with_inverse, with_counts]
    function_tester(rng, F.unique, ref_unique, inputs,
                    func_name=func_name, func_args=func_args,
                    atol_f=0, ctx=ctx, backward=[False])
