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
import numpy as np


def gelu_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes):
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
    c1 = 0.044715
    c3 = 0.134145
    v = np.sqrt(2 / np.pi) * (x0 + c1 * x0 ** 3)
    u = 1 + F.tanh(v)
    t1 = 0.5 * u
    t2 = 0.5 * x0 * (1 - F.tanh(v) ** 2) * \
        np.sqrt(2 / np.pi) * (1 + c3 * x0 ** 2)
    dx0 = dy * (t1 + t2)
    return dx0
