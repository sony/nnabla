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


def weight_normalization_backward(inputs, dim=0, eps=1e-12):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    w = inputs[1]
    g = inputs[2]
    g_shape = g.shape

    # Create inverted norm of w
    sum_axes = list(filter(lambda x: x != dim, range(w.ndim)))
    w_pow = F.pow_scalar(w, 2.0)
    w_sum = F.sum(w_pow, sum_axes, True)
    w_add = F.add_scalar(w_sum, eps)
    w_norm_inv = F.pow_scalar(w_add, -0.5)

    dyw_sum = F.sum(dy * w, sum_axes, True)

    # w.r.t. dw
    g = g.reshape([s if i == dim else 1 for i, s in enumerate(w.shape)])
    dw = (dy - dyw_sum * (w_norm_inv ** 2) * w) * g * w_norm_inv

    # w.r.t. dg
    dg = dyw_sum * w_norm_inv
    dg = dg.reshape(g_shape)

    return dw, dg
