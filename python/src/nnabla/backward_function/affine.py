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
import functools


class AffineDataGrad(LinearDataGrad):

    def __init__(self, ctx, base_axis=1):
        super(AffineDataGrad, self).__init__(ctx)
        self._linear = _F.Affine(ctx, base_axis)


class AffineFilterGrad(LinearFilterGrad):

    def __init__(self, ctx, base_axis=1):
        super(AffineFilterGrad, self).__init__(ctx)
        self._linear = _F.Affine(ctx, base_axis)


def affine_backward(inputs, base_axis=1):
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
    dfx = AffineDataGrad(ctx, base_axis)
    dfw = AffineFilterGrad(ctx, base_axis)
    dfx.xshape = x0.shape
    dfw.wshape = w0.shape

    dx0 = dfx(dy, w0)
    dw0 = dfw(dy, x0)

    if len(inputs) == 4:
        axes = [i for i in range(0, base_axis)]
        db0 = F.sum(dy, axes, keepdims=False)
        return dx0, dw0, db0
    else:
        return dx0, dw0


def affine_data_grad_backward(inputs, base_axis=1):
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
    dfw = AffineFilterGrad(ctx, base_axis)
    dfw.wshape = w0.shape

    gdy = F.affine(gdx, w0, None, base_axis)
    gw0 = dfw(dy, gdx)
    return gdy, gw0


def affine_filter_grad_backward(inputs, base_axis=1):
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
    dfx = AffineDataGrad(ctx, base_axis)
    dfx.xshape = x0.shape

    gdy = F.affine(x0, gdw, None, base_axis)
    gx0 = dfx(dy, gdw)
    return gdy, gx0
