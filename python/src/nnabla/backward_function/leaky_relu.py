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

from .utils import no_grad, get_output


def leaky_relu_backward(inputs, alpha=0.1, inplace=False):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    x0 = get_output(x0, "LeakyReLU") if inplace else x0
    m0 = F.greater_scalar(x0, 0)  # result is same even if inplace or not
    m1 = 1 - m0
    m0 = no_grad(m0)
    m1 = no_grad(m1)
    dx0 = dy * (m0 + alpha * m1)
    return dx0
