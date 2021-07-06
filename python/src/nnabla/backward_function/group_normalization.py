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
from functools import partial
from .tensor_normalization import tensor_normalization_backward


def group_normalization_backward(inputs, num_groups=None, channel_axis=None, batch_axis=(0,), eps=1e-05, no_scale=False, no_bias=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x = inputs[1]
    g = num_groups
    g_idx = 2 if no_bias else 3
    if not no_scale:
        gamma = inputs[g_idx]

    # Original input shape: [B, C, H, W]
    x_shape = list(x.shape)

    # Grouped input shape: [B, num_groups, C / num_groups, H, W]
    xg_shape = x_shape[:channel_axis] + [g, x_shape[channel_axis] // g]
    if channel_axis < len(x_shape) - 1:
        xg_shape += x_shape[channel_axis+1:]

    # Sum operation for broadcast backward
    axes = list(set(range(len(x_shape))) - set([channel_axis]))
    F_sum = partial(F.sum, axis=axes, keepdims=True)

    # w.r.t. x
    # Affine backward
    dyg = dy * gamma if not no_scale else dy
    # Tensor norm backward
    dyg = dyg.reshape(xg_shape)
    xg = x.reshape(xg_shape)

    axes = list(set(range(xg.ndim)) - set(batch_axis + [channel_axis]))
    grads, xn = tensor_normalization_backward([dyg, xg], axes, eps, True, True)

    dx = grads[0].reshape(x_shape)  # Restore shape
    res_grads = (dx,)

    # w.r.t. beta
    if not no_bias:
        # Sum as broadcast backward
        db = F_sum(dy)
        res_grads += (db,)

    # w.r.t. gamma
    if not no_scale:
        # Sum as broadcast backward
        xn = xn.reshape(x_shape)
        dg = F_sum(dy * xn)
        res_grads += (dg,)

    return res_grads
