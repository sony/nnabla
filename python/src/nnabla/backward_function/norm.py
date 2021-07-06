# Copyright 2020,2021 Sony Corporation.
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


import nnabla.functions as F
from .utils import force_list, no_grad


def norm_backward(inputs, p=None, axes=None, keep_dims=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]

    if p is None:
        p = 2.0
    axes = list(range(x0.ndim)) if axes is None else force_list(axes)

    x_abs = F.abs(x0)
    x_pow = F.pow_scalar(x_abs, p)
    x_sum = F.sum(x_pow, axes, keepdims=True)

    # Add axis for mul2
    if not keep_dims:
        shape = list(x0.shape)
        for a in axes:
            shape[a] = 1
        dy = dy.reshape(shape)

    x_sign = no_grad(F.sign(x0))
    dx = dy * x_sum ** (1./p - 1.) * x_abs ** (p - 1.) * x_sign

    return dx
