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

# from nnabla.function import PythonFunction
from .backward_function import LinearFilterGrad


class EmbedFilterGrad(LinearFilterGrad):

    def __init__(self, ctx, base_axis=1):
        super(EmbedFilterGrad, self).__init__(ctx)
        self._linear = _F.Embed(ctx)

    def backward_impl(self, inputs, outputs, propagate_down=[], accum=[]):
        if not propagate_down[0]:
            return

        dy = inputs[0].data
        x0 = inputs[1].data
        dw0 = outputs[0].data

        gdy = inputs[0].grad
        gx0 = inputs[1].grad
        gdw0 = outputs[0].grad

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        vx = inputs_fwd[0].apply(need_grad=False)
        vw = inputs_fwd[1].apply(need_grad=False)
        vy = outputs_fwd[0]

        # w.r.t. x0
        # do nothing since x0 is the index.

        # w.r.t. dy
        vx.data = x0
        vw.data = gdw0
        if accum[0]:
            self._linear.forward(inputs_fwd, outputs_fwd)
            gdy += vy.data
        else:
            vy.data = gdy
            self._linear.forward(inputs_fwd, outputs_fwd)


def embed_backward(inputs):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    w0 = inputs[2]

    ctx = nn.get_current_context()
    dfw = EmbedFilterGrad(ctx)
    dfw.wshape = w0.shape

    dw0 = dfw(dy, x0)
    return None, dw0


def embed_filter_grad_backward(inputs):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    gdw = inputs[0]
    dy = inputs[1]
    x0 = inputs[2]
    gdy = F.embed(x0, gdw)
    return gdy, None
