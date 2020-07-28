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


import numpy as np
import nnabla as nn
import nnabla.functions as F
from .utils import force_list


def mean_backward(inputs, axes=None, keep_dims=False):
    dy = inputs[0]
    x0 = inputs[1]
    axes = [i for i in range(x0.ndim)] if axes is None else force_list(axes)
    n = np.prod([s if i in axes else 1 for i, s in enumerate(x0.shape)])
    if keep_dims:
        dx0 = F.broadcast(dy, x0.shape)
    else:
        shape = [1 if i in axes else s for i, s in enumerate(x0.shape)]
        dx0 = F.broadcast(F.reshape(dy, shape, inplace=False), x0.shape)
    return dx0 / n
