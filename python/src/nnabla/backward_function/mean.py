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
import numpy as np

from .utils import force_list


def mean_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axes=None, keep_dims=False):
    dy = grad_inputs[0]
    x0_shape = input_shapes[0]
    x0_ndim = len(x0_shape)
    axes = [i for i in range(x0_ndim)] if axes is None else force_list(axes)
    n = np.prod([s if i in axes else 1 for i, s in enumerate(x0_shape)])
    if keep_dims:
        dx0 = F.broadcast(dy, x0_shape)
    else:
        shape = [1 if i in axes else s for i, s in enumerate(x0_shape)]
        dx0 = F.broadcast(F.reshape(dy, shape, inplace=False), x0_shape)
    return dx0 / n
