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
import nnabla.function as _F
import nnabla.functions as F
from nnabla.function import PythonFunction


class ConcatenateDataGrad(PythonFunction):

    def __init__(self, ctx, axis=None):
        super(ConcatenateDataGrad, self).__init__(ctx)
        self._func = _F.Concatenate(ctx, axis=axis)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return self._func.args

    def _create_fwd_inputs_outputs(self, inputs, outputs):
        # fwd_inputs: x_1, ..., x_n
        # fwd_outputs: y
        oshape = inputs[0].shape
        inputs_fwd = []
        for xshape in self.xshapes:
            input_fwd = nn.Variable(xshape, need_grad=True)
            inputs_fwd.append(input_fwd)
        outputs_fwd = [nn.Variable(oshape)]
        return inputs_fwd, outputs_fwd

    def min_inputs(self):
        return 1

    def min_outputs(self):
        return len(self.xshapes)

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False

    @property
    def xshapes(self):
        return self._xshapes

    @xshapes.setter
    def xshapes(self, xshapes):
        self._xshapes = xshapes

    def setup_impl(self, inputs, outputs):
        # inputs:  dy
        # outputs: dx_1, ..., dx_n
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        self._func.setup(inputs_fwd, outputs_fwd)
        for i, xshape in enumerate(self.xshapes):
            outputs[i].reset_shape(xshape, True)

    def forward_impl(self, inputs, outputs):
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        dy = inputs[0].data
        dx_list = [outputs[i].data for i in range(len(self.xshapes))]

        vy = outputs_fwd[0]
        vx_list = inputs_fwd

        vy.grad = dy
        for dx, vx in zip(dx_list, vx_list):
            vx.grad = dx
        self._func.backward(inputs_fwd, outputs_fwd, [
                            False] * len(self.xshapes))

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        if not propagate_down[0]:
            return

        gdy = inputs[0].grad
        gdx_list = [output.grad for output in outputs]

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx_list = inputs_fwd
        vy = outputs_fwd[0]

        for gdx, vx in zip(gdx_list, vx_list):
            vx.data = gdx
        if accum[0]:
            self._func.forward(inputs_fwd, outputs_fwd)
            gdy += vy.data
        else:
            vy.data = gdy
            self._func.forward(inputs_fwd, outputs_fwd)


def concatenate_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axis=None):
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
    axis = axis if axis is not None else len(dy.shape) - 1
    ctx = nn.get_current_context()
    df = ConcatenateDataGrad(ctx, axis=axis)
    df.xshapes = input_shapes
    dx0 = df(dy)
    return dx0


def concatenate_data_grad_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axis=None):
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
    gdx = grad_inputs
    dy = inputs[0]
    axis = axis if axis is not None else len(dy.shape) - 1
    gdy = F.concatenate(*gdx, axis=axis)
    return gdy
