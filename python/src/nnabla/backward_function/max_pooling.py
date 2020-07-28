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


def max_pooling_backward(inputs, kernel, stride=None, ignore_border=True, pad=None, channel_last=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    # this is ricky to support inputs of both y = f(x) and gdy = gdf(gdx, x0)
    dy = inputs[0]
    x0 = inputs[-1]
    dx = F.max_pooling_backward(
        dy, x0, kernel, stride, ignore_border, pad, channel_last)
    if len(inputs) == 2:  # dx = df(dy, x)
        return dx
    # ddy = ddf(ddx, dy, x) since x is needed to compute max indices
    else:
        return dx, None
