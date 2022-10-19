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


def weight_normalization_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, dim=0, eps=1e-12):
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
    w = inputs[0]
    g = inputs[1]
    g_shape = g.shape
    dim += w.ndim*(dim < 0)

    # Create inverted norm of w
    sum_axes = list(filter(lambda x: x != dim, range(w.ndim)))
    w_pow = F.pow_scalar(w, 2.0)
    w_sum = F.sum(w_pow, sum_axes, True)
    w_add = F.add_scalar(w_sum, eps)
    w_norm_inv = F.pow_scalar(w_add, -0.5)

    dyw_sum = F.sum(dy * w, sum_axes, True)

    # w.r.t. dw
    g = g.reshape([s if i == dim else 1 for i, s in enumerate(w.shape)])
    dw = (dy - dyw_sum * (w_norm_inv ** 2) * w) * g * w_norm_inv

    # w.r.t. dg
    dg = dyw_sum * w_norm_inv
    dg = dg.reshape(g_shape)

    return dw, dg
