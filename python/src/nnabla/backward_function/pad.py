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
from .backward_function import UnaryDataGrad


class PadDataGrad(UnaryDataGrad):

    def __init__(self, ctx, pad_width, mode='constant', constant_value=0):
        super(PadDataGrad, self).__init__(ctx)
        self._func = _F.Pad(ctx, pad_width, mode, constant_value)


def pad_backward(inputs, pad_width, mode='constant', constant_value=0):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    if mode != "constant":
        raise NotImplementedError(
            "{}_backward (mode!=constant) is not implemented.".format(func['snake_name']))
    dy = inputs[0]
    x0 = inputs[1]
    ctx = nn.get_current_context()
    # constant value is always zero after 1st-order derivative
    df = PadDataGrad(ctx, pad_width, mode, constant_value=0)
    df.xshape = x0.shape
    dx0 = df(dy)
    return dx0


def pad_data_grad_backward(inputs, pad_width, mode='constant', constant_value=0):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    if mode != "constant":
        raise NotImplementedError(
            "{}_backward (mode!=constant) is not implemented.".format(func['snake_name']))
    gdx = inputs[0]
    gdy = F.pad(gdx, pad_width, mode, constant_value=0)
    return gdy
