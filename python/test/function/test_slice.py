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
from nbla_test_utils import list_context, function_tester
from nnabla.testing import assert_allclose

ctxs = list_context('Slice')


def ref_slice(x, start, stop, step):
    s = [slice(start[axis], stop[axis], step[axis])
         for axis in range(len(start))]
    return x[s]


@pytest.mark.parametrize("ctx, fname", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, start, stop, step", [
    ((2, 2), (0, 0), (5, 5), (1, 1)),
    ((2, 2), (0, 0), (2, 2), (1, 1)),
    ((6, 7, 8), (1, 2, 3), (5, 4, 8), (1, 1, 2)),
    ((6, 7, 6, 5), (4, 3, 2, 1), (5, 6, 5, 4), (1, 2, 3, 4)),
    ((7, 6, 5, 4, 3), (5, 4, 3, 2, 1), (6, 6, 5, 4, 2), (1, 2, 3, 2, 1)),
    ((4, 4, 4, 4, 3, 5), (0, 1, 0, 1, 1, 2),
     (2, 4, 2, 4, 2, 5), (2, 1, 2, 1, 1, 1)),
    ((4, 4, 4, 4, 3, 5, 3), (0, 1, 0, 1, 1, 2, 2),
     (2, 4, 2, 4, 2, 4, 3), (2, 1, 2, 1, 1, 1, 1)),
    ((2, 3, 2, 3, 3, 5, 3, 4), (0, 1, 0, 1, 1, 2, 2, 0),
     (2, 3, 2, 3, 2, 4, 3, 4), (2, 1, 2, 1, 1, 1, 1, 2)),
    ((6, 7, 6, 5), (0, 0, 1, 2), (6, -1, -2, -3), (1, 1, 1, 1)),
    ((6, 7, 6, 5), (4, 3, -2, 1), (5, -6, 5, 4), (-1, 2, 3, 4)),
])
def test_slice_forward_backward(seed, inshape, start, stop, step, ctx, fname):
    rng = np.random.RandomState(seed)
    x = rng.randn(*inshape).astype(np.float32)
    function_tester(rng, F.slice, ref_slice, [x], ctx=ctx, func_name=fname,
                    func_args=[start, stop, step], atol_f=1e-4, atol_b=1e-2)


@pytest.mark.parametrize("ctx, fname", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, start, stop, step", [
    ((4, ), [None, ], [None, ], [-1, ]),
    ((4, ), [None, ], [None, ], [None, ]),
    ((5, ), [3, ], [None, ], [-2, ]),
    ((4, 4, 2), [0, None, 0], [4, None, 2], [1, -1, 1]),
    ((4, 4, 2), [-1, None, 0], [None, None, 2], [-1, -1, 1]),
    ((4, ), [0, ], [2, ], [-1, ]),
])
def test_slice_forward_special(seed, inshape, start, stop, step, ctx, fname):
    x_data = np.random.rand(*inshape)
    # Numpy
    s = [slice(start[axis], stop[axis], step[axis])
         for axis in range(len(start))]
    x_data_key = ref_slice(x_data, start, stop, step)

    # NNabla
    with nn.context_scope(ctx):
        x = nn.Variable.from_numpy_array(x_data)
        x_key = F.slice(x, start, stop, step)
        x_key.forward()

    assert_allclose(x_data_key, x_key.d)


@pytest.mark.parametrize("ctx, fname", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, start, stop, step", [
    ((2, 2), (0, 0), (2, 2), (1, 1)),
    ((6, 7, 8), (1, 2, 3), (5, 4, 8), (1, 1, 2)),
    ((6, 7, 6, 5), (4, 3, 2, 1), (5, 6, 5, 4), (1, 2, 3, 4)),
    ((7, 6, 5, 4, 3), (5, 4, 3, 2, 1), (6, 6, 5, 4, 2), (1, 2, 3, 2, 1)),
    # Negative but not empty array, different from test_slice_forward_backward
    ((6, 7, 6, 5), (0, 0, 1, 2), (6, -1, -2, -2), (1, 1, 1, 1)),
    ((6, 7, 6, 5), (5, 0, -2, 1), (4, -6, 5, 4), (-1, 2, 3, 4)),
])
def test_slice_double_backward(seed, inshape, start, stop, step, ctx, fname):
    from nbla_test_utils import backward_function_tester, grad_function_forward_function_output
    from nnabla.backward_function.slice import SliceDataGrad
    rng = np.random.RandomState(seed)
    x = rng.randn(*inshape).astype(np.float32)
    func_args = [start, stop, step]
    # 2nd-order
    backward_function_tester(rng, F.slice, [x], ctx=ctx,
                             func_args=func_args)
    # 3rd-order
    df, y = grad_function_forward_function_output(SliceDataGrad,
                                                  F.slice,
                                                  ctx, [x],
                                                  *func_args)
    df.xshape = x.shape
    ginputs = [rng.randn(*y.shape)]
    backward_function_tester(rng, df,
                             ginputs, func_args=[],
                             ctx=ctx, non_accum_check=True)
