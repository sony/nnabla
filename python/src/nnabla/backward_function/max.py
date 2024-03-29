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

from .utils import no_grad, force_list


def max_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axes=None, keep_dims=False, with_index=False, only_index=False):
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
    # In auto-forward mode, the dynamic clear of inputs[0] and outputs[0] are
    # blocked by Max::auto_grad_depends_input/output_data.
    dy = grad_inputs[0]
    x0 = inputs[0]
    y0 = outputs[0]
    if keep_dims:
        y0 = F.broadcast(y0, x0.shape)
        dy = F.broadcast(dy, x0.shape)
    else:
        axes = [i for i in range(
            x0.ndim)] if axes is None else force_list(axes)
        shape = [1 if i in axes else s for i, s in enumerate(x0.shape)]
        y0 = F.broadcast(F.reshape(y0, shape, inplace=False), x0.shape)
        dy = F.broadcast(F.reshape(dy, shape, inplace=False), x0.shape)
    m0 = F.equal(x0, y0)
    m0 = no_grad(m0)
    dx0 = dy * m0
    if not with_index and not only_index:
        return dx0
    elif with_index:
        return dx0, None
    elif only_index:
        return None
