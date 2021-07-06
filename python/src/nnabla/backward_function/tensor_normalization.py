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


import numpy as np
import nnabla.functions as F
from functools import partial


def tensor_normalization_backward(inputs, axes=(1,), eps=1e-05, no_scale=False, no_bias=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
      Variable: Standardized `x`.
    """
    dy = inputs[0]
    x = inputs[1]
    g_idx = 2 if no_bias else 3
    g = inputs[g_idx] if not no_scale else None  # gamma

    # Prerequisite
    F_sum = partial(F.sum, axis=axes, keepdims=True)
    F_mean = partial(F.mean, axis=axes, keepdims=True)

    # Common factors
    de = np.prod([x.shape[i] for i in axes])  # Denominator
    mean = F_mean(x)
    var = F_mean(x ** 2.00) - mean ** 2.0
    # Normalized x
    xn = (x - mean) / ((var + eps) ** 0.5)
    dxn = dy * g if not no_scale else dy
    # Variance and mean grads
    dvar = F_sum(dxn * (x - mean) * (-0.5) * ((var + eps) ** (-3.0/2.0)))
    dmean = F_sum(dxn * -1 / ((var + eps) ** 0.5))

    # w.r.t. x
    dx = dxn / ((var + eps) ** 0.5) + dvar * 2 * (x-mean) / de + dmean/de

    # w.r.t. beta
    db = F_sum(dy)

    # w.r.t. gamma
    dg = F_sum(dy * xn)

    grads = (dx,)
    if not no_bias:
        grads += (db,)
    if not no_scale:
        grads += (dg,)
    return grads, xn
