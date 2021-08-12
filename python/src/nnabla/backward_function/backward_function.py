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
from nnabla.function import PythonFunction


class UnaryDataGrad(PythonFunction):
    """
    Input is the dy and output is the dx.
    Use the function.backward in the forward_impl.
    Use the function.forward in the backward_impl.
    """

    def __init__(self, ctx):
        super(UnaryDataGrad, self).__init__(ctx)
        self._func = None

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return self._func.args

    def _create_fwd_inputs_outputs(self, inputs, outputs):
        dy = inputs[0].data
        ishape = self.xshape
        oshape = dy.shape
        inputs_fwd = [nn.Variable(ishape, need_grad=True)]
        outputs_fwd = [nn.Variable(oshape)]
        return inputs_fwd, outputs_fwd

    def min_inputs(self):
        return 1

    def min_outputs(self):
        return 1

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return False

    @property
    def xshape(self):
        return self._xshape

    @xshape.setter
    def xshape(self, xshape):
        self._xshape = xshape

    def setup_impl(self, inputs, outputs):
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        self._func.setup(inputs_fwd, outputs_fwd)
        oshape = self.xshape
        outputs[0].reset_shape(oshape, True)

    def forward_impl(self, inputs, outputs):
        dy = inputs[0].data
        dx = outputs[0].data

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx = inputs_fwd[0].apply(need_grad=True)
        vy = outputs_fwd[0]

        vx.grad = dx
        vy.grad = dy
        self._func.backward(inputs_fwd, outputs_fwd, [False])

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        if not propagate_down[0]:
            return

        gdy = inputs[0].grad
        gdx = outputs[0].grad

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx = inputs_fwd[0]
        vy = outputs_fwd[0]

        vx.data = gdx
        if accum[0]:
            self._func.forward(inputs_fwd, outputs_fwd)
            gdy += vy.data
        else:
            vy.data = gdy
            self._func.forward(inputs_fwd, outputs_fwd)


class LinearDataGrad(PythonFunction):

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return self._linear.args

    def min_inputs(self):
        return 1

    def min_outputs(self):
        return 1

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return True

    @property
    def xshape(self):
        return self._xshape

    @xshape.setter
    def xshape(self, xshape):
        self._xshape = xshape

    def _create_fwd_inputs_outputs(self, inputs, outputs):
        dy = inputs[0].data
        w0 = inputs[1].data
        ishape = self.xshape
        wshape = w0.shape
        oshape = dy.shape
        inputs_fwd = [nn.Variable(ishape, need_grad=True),
                      nn.Variable(wshape, need_grad=True)]
        outputs_fwd = [nn.Variable(oshape)]
        return inputs_fwd, outputs_fwd

    def setup_impl(self, inputs, outputs):
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        self._linear.setup(inputs_fwd, outputs_fwd)
        oshape = self.xshape
        outputs[0].reset_shape(oshape, True)

    def forward_impl(self, inputs, outputs):
        dy = inputs[0].data
        w0 = inputs[1].data
        dx = outputs[0].data

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx = inputs_fwd[0].apply(need_grad=True)
        vw = inputs_fwd[1].apply(need_grad=False)
        vy = outputs_fwd[0]

        vx.grad = dx
        vw.data = w0
        vy.grad = dy
        self._linear.backward(inputs_fwd, outputs_fwd, [False, False])

    def backward_impl(self, inputs, outputs, propagate_down=[], accum=[]):
        dy = inputs[0].data
        w0 = inputs[1].data
        dx = outputs[0].data

        gdy = inputs[0].grad
        gw0 = inputs[1].grad
        gdx = outputs[0].grad

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx = inputs_fwd[0].apply(need_grad=False)
        vw = inputs_fwd[1].apply(need_grad=propagate_down[1])
        vy = outputs_fwd[0]

        # w.r.t. w0
        if propagate_down[1]:
            vx.data = gdx
            vy.grad = dy
            vw.grad = gw0
            self._linear.backward(inputs_fwd, outputs_fwd, [False, accum[1]])

        # w.r.t. dy
        if propagate_down[0]:
            vx.data = gdx
            vw.data = w0
            if accum[0]:
                self._linear.forward(inputs_fwd, outputs_fwd)
                gdy += vy.data
            else:
                vy.data = gdy
                self._linear.forward(inputs_fwd, outputs_fwd)


class LinearFilterGrad(PythonFunction):

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return self._linear.args

    def min_inputs(self):
        return 1

    def min_outputs(self):
        return 1

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return True

    @property
    def wshape(self):
        return self._wshape

    @wshape.setter
    def wshape(self, wshape):
        self._wshape = wshape

    def _create_fwd_inputs_outputs(self, inputs, outputs):
        dy = inputs[0].data
        x0 = inputs[1].data
        ishape = x0.shape
        wshape = self.wshape
        oshape = dy.shape
        inputs_fwd = [nn.Variable(ishape, need_grad=True),
                      nn.Variable(wshape, need_grad=True)]
        outputs_fwd = [nn.Variable(oshape)]
        return inputs_fwd, outputs_fwd

    def setup_impl(self, inputs, outputs):
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        self._linear.setup(inputs_fwd, outputs_fwd)
        oshape = self.wshape
        outputs[0].reset_shape(oshape, True)

    def forward_impl(self, inputs, outputs):
        dy = inputs[0].data
        x0 = inputs[1].data
        dw = outputs[0].data

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx = inputs_fwd[0].apply(need_grad=False)
        vw = inputs_fwd[1].apply(need_grad=True)
        vy = outputs_fwd[0]

        vx.data = x0
        vy.grad = dy
        vw.grad = dw
        self._linear.backward(inputs_fwd, outputs_fwd, [False, False])

    def backward_impl(self, inputs, outputs, propagate_down=[], accum=[]):
        dy = inputs[0].data
        x0 = inputs[1].data
        dw = outputs[0].data

        gdy = inputs[0].grad
        gx0 = inputs[1].grad
        gdw = outputs[0].grad

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx = inputs_fwd[0].apply(need_grad=True)
        vw = inputs_fwd[1].apply(need_grad=False)
        vy = outputs_fwd[0]

        # w.r.t. x0
        if propagate_down[1]:
            vx.grad = gx0
            vw.data = gdw
            vy.grad = dy
            self._linear.backward(inputs_fwd, outputs_fwd, [accum[1], False])

        # w.r.t. dy
        if propagate_down[0]:
            vx.data = x0
            vw.data = gdw
            if accum[0]:
                self._linear.forward(inputs_fwd, outputs_fwd)
                gdy += vy.data
            else:
                vy.data = gdy
                self._linear.forward(inputs_fwd, outputs_fwd)
