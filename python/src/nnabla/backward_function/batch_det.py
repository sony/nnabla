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
from .utils import no_grad, get_output


def batch_det_backward(inputs):
    """
    Args:
      inputs (list of nn.Variable): Incomming grads/inputs to/of the forward function.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    dy = inputs[0]
    x0 = inputs[1]
    x0_inv_T = F.transpose(F.batch_inv(x0), [0, 2, 1])
    b = x0.shape[0]
    dy = F.reshape(dy, [b, 1, 1])
    y0 = get_output(x0, "BatchDet")
    y0 = F.reshape(y0, [b, 1, 1], inplace=False)
    dx0 = dy * y0 * x0_inv_T
    return dx0
