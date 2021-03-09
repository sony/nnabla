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


import nnabla.functions as F
from .utils import no_grad, force_list


def prod_backward(inputs, axes=None, keep_dims=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    axes = [i for i in range(x0.ndim)] if axes is None else force_list(axes)
    y0 = F.prod(x0, axes, keep_dims)
    if keep_dims:
        dx0 = F.broadcast(dy * y0 / x0, x0.shape)
    else:
        shape = [1 if i in axes else s for i, s in enumerate(x0.shape)]
        dx0 = F.broadcast(F.reshape(dy * y0, shape) / x0, x0.shape)
    return dx0
