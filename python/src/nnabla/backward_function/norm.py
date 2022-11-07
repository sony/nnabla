# Copyright 2020,2021 Sony Corporation.
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


import nnabla.functions as F
from .utils import force_list, no_grad


def norm_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, p=None, axes=None, keep_dims=False):
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
    dy = grad_inputs[0]
    x0 = inputs[0]

    if p is None:
        p = 2.0
    axes = list(range(x0.ndim)) if axes is None else force_list(axes)

    x_abs = F.abs(x0)
    x_pow = F.pow_scalar(x_abs, p)
    x_sum = F.sum(x_pow, axes, keepdims=True)

    # Add axis for mul2
    if not keep_dims:
        shape = list(x0.shape)
        for a in axes:
            shape[a] = 1
        dy = dy.reshape(shape)

    x_sign = no_grad(F.sign(x0))
    dx = dy * x_sum ** (1./p - 1.) * x_abs ** (p - 1.) * x_sign

    return dx
