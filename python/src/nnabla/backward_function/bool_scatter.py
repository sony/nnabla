# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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
from .utils import no_grad


def bool_scatter_backward(inputs):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    m0 = inputs[2]
    o0 = inputs[3] if len(inputs) == 4 else None

    dx = F.bool_gather(dy, m0)
    dm = None

    if o0 is None:
        return dx, dm
    else:
        m1 = F.equal_scalar(m0, 0)
        m1 = F.reshape(m1, m1.shape + (1, ) * (dy.ndim - m1.ndim))
        m1 = F.broadcast(m1, dy.shape)
        m1 = no_grad(m1)
        do = dy * m1
        return dx, dm, do
