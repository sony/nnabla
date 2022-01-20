# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from .utils import no_grad, create_slice


def celu_backward(inputs, alpha=1.0, axis=1):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]

    fstart, fstop, fstep = create_slice(dy.shape, axis, True)
    bstart, bstop, bstep = create_slice(dy.shape, axis, False)
    dy0 = F.slice(dy, fstart, fstop, fstep)
    dy1 = F.slice(dy, bstart, bstop, bstep)
    aep = alpha * F.exp(x0)
    aen = alpha * F.exp(-x0)

    m0 = F.greater_scalar(x0, 0)
    m1 = 1 - m0
    m0 = no_grad(m0)
    m1 = no_grad(m1)
    dx00 = dy0 * (m0 + aep * m1)
    dx01 = dy1 * (m1 + aen * m0)

    dx = dx00 - dx01
    return dx
