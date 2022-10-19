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


import nnabla.functions as F

from .utils import no_grad, positive_axis


def categorical_cross_entropy_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axis=None):
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
    t0 = inputs[1]

    D = len(x0.shape)
    axis = positive_axis(axis, D)
    c0 = x0.shape[axis]
    t0_shape = [s for s in t0.shape if s != 1]
    u0 = F.reshape(t0, (-1, 1), inplace=False)
    u1 = F.one_hot(u0, (c0, ))
    to = F.reshape(u1, t0_shape + [c0, ])
    t0 = no_grad(to)
    if axis != len(to.shape) - 1:
        oaxes = [i for i in range(len(t0_shape))]
        taxes = oaxes[:axis] + [to.ndim - 1] + oaxes[axis:]
        to = F.transpose(to, taxes)
    dx0 = -dy * to / x0
    return dx0, None
