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

from .backward_function import LinearDataGrad, LinearFilterGrad


class DeconvolutionDataGrad(LinearDataGrad):

    def __init__(self, ctx, base_axis=1, pad=None, stride=None, dilation=None, group=1, channel_last=False, output_padding=None):
        super(DeconvolutionDataGrad, self).__init__(ctx)
        self._linear = _F.Deconvolution(
            ctx, base_axis, pad, stride, dilation, group, channel_last, output_padding)


class DeconvolutionFilterGrad(LinearFilterGrad):

    def __init__(self, ctx, base_axis=1, pad=None, stride=None, dilation=None, group=1, channel_last=False, output_padding=None):
        super(DeconvolutionFilterGrad, self).__init__(ctx)
        self.base_axis = base_axis
        self._linear = _F.Deconvolution(
            ctx, base_axis, pad, stride, dilation, group, channel_last, output_padding)


def deconvolution_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, base_axis=1, pad=None, stride=None, dilation=None, group=1, channel_last=False, output_padding=None):
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
    w0 = inputs[1]

    base_axis += x0.ndim*(base_axis < 0)
    # base_axis += grad_inputs[0].ndim*(base_axis < 0)

    ctx = nn.get_current_context()
    dfx = DeconvolutionDataGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last, output_padding)
    dfw = DeconvolutionFilterGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last, output_padding)
    dfx.xshape = x0.shape
    dfw.wshape = w0.shape

    dx0 = dfx(dy, w0)
    dw0 = dfw(dy, x0)
    axes = [i for i in range(dy.ndim, base_axis)]
    db0 = F.sum(dy, axes, keepdims=False) if len(inputs) == 3 else None

    if len(inputs) == 3:
        if channel_last:
            axes = [i for i in range(dy.ndim - 1)]
        else:
            axes = [i for i in range(0, base_axis)] + \
                                     [i for i in range(base_axis + 1, dy.ndim)]
        db0 = F.sum(dy, axes, keepdims=False) if len(inputs) == 3 else None
        return dx0, dw0, db0
    else:
        return dx0, dw0


def deconvolution_data_grad_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, base_axis=1, pad=None, stride=None,
                                     dilation=None, group=1, channel_last=False, output_padding=None):
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
    w0 = inputs[1]

    ctx = nn.get_current_context()
    dfw = DeconvolutionFilterGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last, output_padding)
    dfw.wshape = w0.shape

    gdy = F.deconvolution(gdx, w0, None, base_axis, pad,
                          stride, dilation, group, channel_last, output_padding)
    gw0 = dfw(dy, gdx)
    return gdy, gw0


def deconvolution_filter_grad_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, base_axis=1, pad=None, stride=None,
                                       dilation=None, group=1, channel_last=False, output_padding=None):
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
    gdw = grad_inputs[0]
    dy = inputs[0]
    x0 = inputs[1]

    ctx = nn.get_current_context()
    dfx = DeconvolutionDataGrad(
        ctx, base_axis, pad, stride, dilation, group, channel_last, output_padding)
    dfx.xshape = x0.shape

    gdy = F.deconvolution(x0, gdw, None, base_axis, pad,
                          stride, dilation, group, channel_last, output_padding)
    gx0 = dfx(dy, gdw)
    return gdy, gx0
