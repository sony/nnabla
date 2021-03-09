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

import nnabla as nn
import nnabla.functions as F
import nnabla.functions as F
from .utils import no_grad


def _sum(dx, x):
    axes = [a for a in range(x.ndim - 2) if dx.shape[a] != x.shape[a]]
    dx = F.sum(dx, axes, keepdims=True) if axes != [] else dx
    return dx


def batch_matmul_backward(inputs, transpose_a=False, transpose_b=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dc = inputs[0]
    a = inputs[1]
    b = inputs[2]
    if (transpose_a, transpose_b) == (True, True):
        da = F.batch_matmul(b, dc, True, True)
        db = F.batch_matmul(dc, a, True, True)
    elif (transpose_a, transpose_b) == (True, False):
        da = F.batch_matmul(b, dc, False, True)
        db = F.batch_matmul(a, dc, False, False)
    elif (transpose_a, transpose_b) == (False, True):
        da = F.batch_matmul(dc, b, False, False)
        db = F.batch_matmul(dc, a, True, False)
    elif (transpose_a, transpose_b) == (False, False):
        da = F.batch_matmul(dc, b, False, True)
        db = F.batch_matmul(a, dc, True, False)
    da = _sum(da, a)
    db = _sum(db, b)
    return da, db
