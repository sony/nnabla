# Copyright 2019,2020,2021 Sony Corporation.
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

import nnabla as nn
import nnabla.functions as F


def _sum(dx, x):
    axes = [a for a in range(x.ndim - 2) if dx.shape[a] != x.shape[a]]
    dx = F.sum(dx, axes, keepdims=True) if axes != [] else dx
    return dx


def batch_matmul_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, transpose_a=False, transpose_b=False):
    """
    Args:
      grad_inputs (list of :obj:`nnabla.Variable`): Propagated grads to this backward function.
      inputs (list of :obj:`nnabla.Variable` and None): Input Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      input_shapes (list of tuple of :obj:`int`): Input shapes of the forward function.
          The shapes of the inputs in which None is set can be passed.
      outputs (list of :obj:`nnabla.Variable` and None): Output Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      output_shapes (list of tuple of :obj:`int`): Output shapes of the forward function.
          The shapes of the outputs in which None is set can be passed.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dc = grad_inputs[0]
    a = inputs[0]
    b = inputs[1]
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
