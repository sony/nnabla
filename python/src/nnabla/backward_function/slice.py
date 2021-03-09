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
from .utils import force_list


class SliceDataGrad(UnaryDataGrad):

    @property
    def name(self):
        return 'SliceDataGrad'

    def __init__(self, ctx, start=None, stop=None, step=None):
        super(SliceDataGrad, self).__init__(ctx)
        self._func = _F.Slice(ctx, start, stop, step)


def slice_backward(inputs, start=None, stop=None, step=None):
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
    df = SliceDataGrad(ctx, start, stop, step)
    df.xshape = x0.shape
    dx0 = df(dy)
    return dx0


def slice_data_grad_backward(inputs, start=None, stop=None, step=None):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    gdx = inputs[0]
    gdy = F.slice(gdx, start, stop, step)
    return gdy
