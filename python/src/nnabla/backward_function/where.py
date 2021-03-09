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
from .utils import no_grad


def where_backward(inputs):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    cd = inputs[1]
    xt = inputs[2]
    xf = inputs[3]
    c1 = F.constant(1, xt.shape)
    c0 = F.constant(0, xf.shape)
    m0 = F.where(cd, c1, c0)
    m1 = 1 - m0
    m0 = no_grad(m0)
    m1 = no_grad(m1)
    dx0 = dy * m0
    dx1 = dy * m1
    return None, dx0, dx1
