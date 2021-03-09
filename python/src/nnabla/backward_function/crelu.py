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
from .utils import no_grad, create_slice


def crelu_backward(inputs, axis=1):
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

    m0 = F.greater_scalar(x0, 0)
    m1 = 1 - m0
    m0 = no_grad(m0)
    m1 = no_grad(m1)
    dx = dy0 * m0 - dy1 * m1
    return dx
