# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import nnabla as nn
import nnabla.functions as F
import nnabla.function as _F
from .utils import no_grad
from .backward_function import LinearDataGrad, LinearFilterGrad


class ConvolutionDataGrad(LinearDataGrad):

    def __init__(self, ctx, base_axis=1, pad=None, stride=None, dilation=None, group=1, channel_last=False):
        super(ConvolutionDataGrad, self).__init__(ctx)
        self._linear = _F.Convolution(
            ctx, base_axis, pad, stride, dilation, group, channel_last)


class ConvolutionFilterGrad(LinearFilterGrad):

    def __init__(self, ctx, base_axis=1, pad=None, stride=None, dilation=None, group=1, channel_last=False):
        super(ConvolutionFilterGrad, self).__init__(ctx)
        self.base_axis = base_axis
        self._linear = _F.Convolution(
            ctx, base_axis, pad, stride, dilation, group, channel_last)


def convolution_backward(inputs, base_axis=1, pad=None, stride=None,
                         dilation=None, group=1, channel_last=False):
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
    dfx = ConvolutionDataGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last)
    dfw = ConvolutionFilterGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last)
    dfx.xshape = x0.shape
    dfw.wshape = w0.shape

    dx0 = dfx(dy, w0)
    dw0 = dfw(dy, x0)

    if len(inputs) == 4:
        if channel_last:
            axes = [i for i in range(dy.ndim - 1)]
        else:
            axes = [i for i in range(0, base_axis)] + \
                                     [i for i in range(base_axis + 1, dy.ndim)]
        db0 = F.sum(dy, axes, keepdims=False) if len(inputs) == 4 else None
        return dx0, dw0, db0
    else:
        return dx0, dw0


def convolution_data_grad_backward(inputs, base_axis=1, pad=None, stride=None,
                                   dilation=None, group=1, channel_last=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    gdx = inputs[0]
    dy = inputs[1]
    w0 = inputs[2]

    ctx = nn.get_current_context()
    dfw = ConvolutionFilterGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last)
    dfw.wshape = w0.shape

    gdy = F.convolution(gdx, w0, None, base_axis, pad,
                        stride, dilation, group, channel_last)
    gw0 = dfw(dy, gdx)
    return gdy, gw0


def convolution_filter_grad_backward(inputs, base_axis=1, pad=None, stride=None,
                                     dilation=None, group=1, channel_last=False):
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

    ctx = nn.get_current_context()
    dfx = ConvolutionDataGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last)
    dfx.xshape = x0.shape

    gdy = F.convolution(x0, gdw, None, base_axis, pad,
                        stride, dilation, group, channel_last)
    gx0 = dfx(dy, gdw)
    return gdy, gx0
