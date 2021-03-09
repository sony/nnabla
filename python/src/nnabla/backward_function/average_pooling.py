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


class AveragePoolingDataGrad(UnaryDataGrad):

    def __init__(self, ctx, kernel, stride=None, ignore_border=True, pad=None,
                 channel_last=False, including_pad=True):
        super(AveragePoolingDataGrad, self).__init__(ctx)
        self._func = _F.AveragePooling(ctx, kernel, stride, ignore_border, pad,
                                       channel_last, including_pad)


def average_pooling_backward(inputs, kernel, stride=None,
                             ignore_border=True, pad=None,
                             channel_last=False, including_pad=True):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    ctx = nn.get_current_context()
    df = AveragePoolingDataGrad(ctx, kernel, stride, ignore_border, pad,
                                channel_last, including_pad)
    df.xshape = x0.shape
    dx0 = df(dy)
    return dx0


def average_pooling_data_grad_backward(inputs, kernel, stride=None,
                                       ignore_border=True, pad=None,
                                       channel_last=False, including_pad=True):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    gdx = inputs[0]
    gdy = F.average_pooling(gdx, kernel, stride, ignore_border, pad,
                            channel_last, including_pad)
    return gdy
