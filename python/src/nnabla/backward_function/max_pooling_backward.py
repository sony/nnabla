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
from nnabla.function import PythonFunction


class MaxPoolingBackwardDataGrad(PythonFunction):
    """
    Input is the dy and output is the dx.
    Use the function.backward in the forward_impl.
    Use the function.forward in the backward_impl.
    """

    def __init__(self, ctx, kernel, stride=None,
                 ignore_border=True, pad=None, channel_last=False):
        super(MaxPoolingBackwardDataGrad, self).__init__(ctx)
        self._func = _F.MaxPoolingBackward(ctx, kernel, stride,
                                           ignore_border, pad, channel_last)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return self._func.args

    def _create_fwd_inputs_outputs(self, inputs, outputs):
        dx = inputs[0].data
        ishape = self.yshape
        oshape = dx.shape
        inputs_fwd = [nn.Variable(ishape, need_grad=True),
                      nn.Variable(oshape, need_grad=False)]
        outputs_fwd = [nn.Variable(oshape)]
        # inputs_fwd[0]:  dy
        # inputs_fwd[1]:  x0
        # outputs_fwd[0]: dx
        return inputs_fwd, outputs_fwd

    def min_inputs(self):
        return 2

    def min_outputs(self):
        return 1

    @property
    def yshape(self):
        return self._yshape

    @yshape.setter
    def yshape(self, yshape):
        self._yshape = yshape

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return True

    def setup_impl(self, inputs, outputs):
        # inputs[0]:  gdx
        # inputs[1]:  x0
        # outputs[0]: gdy

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        self._func.setup(inputs_fwd, outputs_fwd)
        oshape = self.yshape
        outputs[0].reset_shape(oshape, True)

    def forward_impl(self, inputs, outputs):
        dx = inputs[0].data
        x0 = inputs[1].data
        dy = outputs[0].data

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx0 = inputs_fwd[0].apply(need_grad=True)
        vx1 = inputs_fwd[1].apply(need_grad=True)
        vy0 = outputs_fwd[0]

        vx0.grad = dy
        vx1.data = x0
        vy0.grad = dx
        self._func.backward(inputs_fwd, outputs_fwd, [False])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        if not propagate_down[0]:
            return

        gdx = inputs[0].grad
        x0 = inputs[1].data
        gdy = outputs[0].grad

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx0 = inputs_fwd[0]
        vx1 = inputs_fwd[1]
        vy0 = outputs_fwd[0]

        vx0.data = gdy
        vx1.data = x0

        if accum[0]:
            self._func.forward(inputs_fwd, outputs_fwd)
            gdx += vy0.data
        else:
            vy0.data = gdx
            self._func.forward(inputs_fwd, outputs_fwd)


def max_pooling_backward_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, kernel, stride=None,
                                  ignore_border=True, pad=None, channel_last=False):
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
    gdx = grad_inputs[0]
    dy = inputs[0]
    x0 = inputs[1]
    ctx = nn.get_current_context()
    df = MaxPoolingBackwardDataGrad(
        ctx, kernel, stride, ignore_border, pad, channel_last)
    df.yshape = dy.shape
    gdy = df(gdx, x0)
    return gdy, None
