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
from .utils import no_grad, positive_axis


def softmax_cross_entropy_backward(inputs, axis=None):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    t0 = inputs[2]

    D = len(x0.shape)
    axis = positive_axis(axis, D)
    c0 = x0.shape[axis]
    t0_shape = [s for s in t0.shape if s != 1]
    u0 = F.reshape(t0, (-1, 1), inplace=False)
    u1 = F.one_hot(u0, (c0, ))
    to = F.reshape(u1, t0_shape + [c0, ])
    t0 = no_grad(to)
    if axis != len(to.shape) - 1:
        oaxes = [i for i in range(len(t0_shape))]
        taxes = oaxes[:axis] + [to.ndim - 1] + oaxes[axis:]
        to = F.transpose(to, taxes)
    dx0 = dy * (F.softmax(x0, axis=axis) - to)
    return dx0, None
