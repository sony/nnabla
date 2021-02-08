# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('Gather')


def gather(x, indices, axis, batch_dims):
    xshape = x.shape
    ishape = indices.shape
    bshape = xshape[:batch_dims]
    samples = np.prod(bshape).astype(np.int)
    x = x.reshape((samples, ) + xshape[batch_dims:])
    indices = indices.reshape((samples, ) + ishape[batch_dims:])
    y_list = []
    for b in range(samples):
        y = np.take(x[b], indices[b], axis=axis - batch_dims)
        y_list.append(y)
    y = np.stack(y_list) if samples != 1 else y_list[0]
    y = y.reshape(bshape + y.shape[1:]) if samples != 1 else y
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("xshape, ishape, axis, batch_dims", [
   # batch_dims = 0
   ((2, 3, 4), (2, ), 0, 0),
   ((2, 3, 4), (2, ), 1, 0),
   ((2, 3, 4), (2, ), 2, 0),
   ((2, 3, 4), (2, ), -1, 0),
   ((2, 3, 4), (2, ), -2, 0),
   ((2, 3, 4), (2, ), -3, 0),
   ((2, 3, 4), (2, 2), 0, 0),
   ((2, 3, 4), (2, 2), 1, 0),
   ((2, 3, 4), (2, 2), 2, 0),
   ((2, 3, 4), (2, 2), -1, 0),
   ((2, 3, 4), (2, 2), -2, 0),
   ((2, 3, 4), (2, 2), -3, 0),
   ((2, 3, 4), (1, 2, 2), 0, 0),
   ((2, 3, 4), (1, 2, 2), 1, 0),
   ((2, 3, 4), (1, 2, 2), 2, 0),
   ((2, 3, 4), (1, 2, 2, 1), 0, 0),
   ((2, 3, 4), (1, 2, 2, 1), 1, 0),
   ((2, 3, 4), (1, 2, 2, 1), 2, 0),
   ((2, 3, 4), (1, 2, 2, 1), -1, 0),
   ((2, 3, 4), (1, 2, 2, 1), -2, 0),
   ((2, 3, 4), (1, 2, 2, 1), -3, 0),
   # batch_dims = 1
   ((2, 3, 4), (2, 2, 2, 1), 1, 1),
   ((2, 3, 4), (2, 2, 2, 1), 2, 1),
   # batch_dims = 2
   ((2, 3, 4, 5), (2, 3, 2, 1), 2, 2),
   ((2, 3, 4, 5), (2, 3, 2, 1), 3, 2),
])
def test_forward_backward(seed, xshape, ishape, axis, batch_dims, ctx, func_name):
    rng = np.random.RandomState(seed)
    x = rng.randn(*xshape).astype(np.float32)
    indices = rng.randint(0, xshape[axis], ishape)
    inputs = [x, indices]
    func_args = [axis, batch_dims]
    function_tester(rng, F.gather, gather, inputs, func_name=func_name, func_args=func_args,
                    ctx=ctx, backward=[True, False], disable_half_test=False)
