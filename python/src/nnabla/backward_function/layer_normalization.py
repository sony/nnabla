# Copyright 2021 Sony Corporation.
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
from .tensor_normalization import tensor_normalization_backward


def layer_normalization_backward(inputs, batch_axis=(0,), eps=1e-05, no_scale=False, no_bias=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x = inputs[1]
    g_idx = 2 if no_bias else 3
    if not no_scale:
        gamma = inputs[g_idx]

    # w.r.t. x
    axes = list(set(range(x.ndim)) - set(batch_axis))
    dy_tn = dy * gamma if not no_scale else dy
    grads, xn = tensor_normalization_backward(
        [dy_tn, x], axes, eps, True, True)
    dx = grads[0]
    res_grads = (dx,)

    # w.r.t. beta
    if not no_bias:
        db = F.sum(dy, axis=batch_axis, keepdims=True)
        res_grads += (db,)

    # w.r.t. gamma
    if not no_scale:
        dg = F.sum(dy * xn, axis=batch_axis, keepdims=True)
        res_grads += (dg,)

    return res_grads
