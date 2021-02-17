# Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('Sort')


def ref_sort_fw(x, axis, reverse, with_index, only_index):
    y = np.sort(x, axis)
    i = np.argsort(x, axis)
    if reverse:
        y = np.flip(y, axis)
        i = np.flip(i, axis)
    if only_index:
        return i
    if with_index:
        return y, i
    return y


def ref_sort_bw(*args, **kwargs):
    if len(args) > 6:
        args = args[0:2] + args[3:]
    x, g, axis, reverse, with_index, only_index = args
    # Move sort axis to last and reshape into two dimensions
    xx = np.moveaxis(x, axis, -1).reshape(-1, x.shape[axis])
    gg = np.moveaxis(g, axis, -1).reshape(-1, x.shape[axis])
    # Argsort the last dimension in normal or reverse order
    ii = np.argsort(xx) if not reverse else np.flip(np.argsort(xx), -1)
    # Sort gradient vector over outer dimension
    for k in range(ii.shape[0]):
        gg[k] = gg[k, ii[k]]
    # Restore full shape but with sort axis at last
    gg = gg.reshape(np.moveaxis(g, axis, -1).shape)
    # Move the sort axis back to original position
    gg = np.moveaxis(gg, -1, axis)
    # The function tester expects a flattened gradient.
    return gg.flatten()


@pytest.mark.parametrize("ctx, fname", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("with_index", [False, True])
@pytest.mark.parametrize("only_index", [False, True])
@pytest.mark.parametrize("ishape, axis", [
    ((2, 3, 4), -1), ((2, 3, 4), -2), ((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), 2),
    ((100,), 0), ((1, 100), -1), ((1, 100), 0), ((2, 3, 4, 3, 2), 2),
])
def test_forward_backward(seed, ishape, axis, reverse, with_index, only_index,
                          ctx, fname):
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*ishape).astype(np.float32)]
    function_tester(rng, F.sort, ref_sort_fw, inputs, ctx=ctx, func_name=fname,
                    func_args=[axis, reverse, with_index, only_index],
                    ref_grad=ref_sort_bw)


@pytest.mark.parametrize("ctx, fname", ctxs)
@pytest.mark.parametrize("reverse", [False, True])
def test_equal_values(ctx, fname, reverse):
    with nn.context_scope(ctx), nn.auto_forward(True):
        x = nn.Variable.from_numpy_array([2, 3, 3, 4, 2])
        y, i = F.sort(x, reverse=reverse, with_index=True)
        assert all(y.d == ([4, 3, 3, 2, 2] if reverse else [2, 2, 3, 3, 4]))
        assert all(i.d == ([3, 1, 2, 0, 4] if reverse else [0, 4, 1, 2, 3]))
