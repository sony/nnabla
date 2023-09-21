# Copyright 2023 Sony Group Corporation.
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
from nnabla.testing import assert_allclose

ctxs = list_context('Unique')


def force_tuple(x):
    if isinstance(x, nn.Variable):
        return (x,)
    return x


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
    rng = np.random.RandomState(seed)
    input = rng.randint(-10, 10, x_shape).astype(np.float32)
    vinput = nn.Variable.from_numpy_array(input)
    func_args = [flatten, axis, sorted, with_index, with_inverse, with_counts]
    with nn.context_scope(ctx), nn.auto_forward():
        o = F.unique(vinput, *func_args)
        o = force_tuple(o)
    r = ref_unique(input, *func_args)
    assert len(o) == len(r)
    for act, ref in zip(o, r):
        assert_allclose(act.d, ref)
        assert func_name == act.parent.name
