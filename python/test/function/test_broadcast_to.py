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
import nnabla.functions as F

from nbla_test_utils import (
    function_tester,
    list_ctx_and_func_name)


def copying_to_leaf(x, y, axis):
    return (len(x.shape) - len(y.shape) - axis) == 0


def ref_broadcast_to(x, y, axis):
    if axis < 0 or copying_to_leaf(x, y, axis):
        # Copy data to leaf
        return np.ones(x.shape) * y
    # Copy data from specified axis
    xs = len(x.shape)
    ys = len(y.shape)
    if xs == 2:
        t = y[:, np.newaxis]
        t.transpose()
        return np.broadcast_to(t, x.shape)
    elif xs == 3:
        if ys == 1:
            if axis == 0:
                t = y[:, np.newaxis, np.newaxis]
                t.transpose()
                return np.broadcast_to(t, x.shape)
            elif axis == 1:
                t = y[np.newaxis, :, np.newaxis]
                t.transpose()
                return np.broadcast_to(t, x.shape)
        elif ys == 2:
            if axis == 0:
                t = y[:, :, np.newaxis]
                return np.broadcast_to(t, x.shape)
    elif xs == 4:
        if ys == 1:
            if axis == 0:
                t = y[:, np.newaxis, np.newaxis, np.newaxis]
                t.transpose()
                return np.broadcast_to(t, x.shape)
            elif axis == 1:
                t = y[np.newaxis, :, np.newaxis, np.newaxis]
                t.transpose()
                return np.broadcast_to(t, x.shape)
            elif axis == 2:
                t = y[np.newaxis, np.newaxis, :, np.newaxis]
                t.transpose()
                return np.broadcast_to(t, x.shape)
        elif ys == 2:
            if axis == 0:
                t = y[:, :, np.newaxis, np.newaxis]
                return np.broadcast_to(t, x.shape)
            elif axis == 1:
                t = y[np.newaxis, :, :, np.newaxis]
                return np.broadcast_to(t, x.shape)
        elif ys == 3:
            if axis == 0:
                t = y[:, :, :, np.newaxis]
                return np.broadcast_to(t, x.shape)


PARAMS = [
    ((2, 3), (2), 0),
    ((2, 3), (3), 1),
    ((2, 3, 4), (2), 0),
    ((2, 3, 4), (3), 1),
    ((2, 3, 4), (4), 2),
    ((2, 3, 4), (2, 3), 0),
    ((2, 3, 4), (3, 4), 1),
    ((2, 3, 4, 5), (2), 0),
    ((2, 3, 4, 5), (3), 1),
    ((2, 3, 4, 5), (4), 2),
    ((2, 3, 4, 5), (5), 3),
    ((2, 3, 4, 5), (2, 3), 0),
    ((2, 3, 4, 5), (3, 4), 1),
    ((2, 3, 4, 5), (4, 5), 2),
    ((2, 3, 4, 5), (2, 3, 4), 0),
    ((2, 3, 4, 5), (3, 4, 5), 1),
    ((2, 3, 4, 5), (5), -1),
    ((2, 3, 4, 5), (4, 5), -1),
    ((2, 3, 4, 5), (3, 4, 5), -1),
    ((2, 3, 4, 5), (2, 3, 4, 5), -1),
    ((2, 3, 4, 5), (2, 3, 4, 5), -2)
]


@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("fname, ctx, func_name", list_ctx_and_func_name(['broadcast_to']))
@pytest.mark.parametrize("xs, ys, axis", PARAMS)
def test_broadcast_to_forward(xs, ys, axis, seed, fname, ctx, func_name):
    rng = np.random.RandomState(seed)
    ref_func = eval('ref_' + fname)
    func = getattr(F, fname)
    inputs = [rng.random_sample(xs), rng.random_sample(ys)]
    function_tester(rng, func, ref_func, inputs, [axis],
                    backward=[False, False],
                    ctx=ctx, func_name=func_name)
