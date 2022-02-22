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

import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.initializer import (
    UniformInitializer)

from .module import Module


class Embed(Module):
    """ Embed.

    Embed slices a matrix/tensor with indexing array/tensor. Weights are initialized with :obj:`nnabla.initializer.UniformInitializer` within the range of :math:`-\\sqrt{3}` and :math:`\\sqrt{3}`.

    Args:
        x(~nnabla.Variable): [Integer] Indices with shape :math:`(I_0, ..., I_N)`
        n_inputs : number of possible inputs, words or vocabraries
        n_features : number of embedding features
        fix_parameters (bool): When set to `True`, the embedding weight matrix
            will not be updated.

    Returns:
        ~nnabla.Variable: Output with shape :math:`(I_0, ..., I_N, W_1, ..., W_M)`
    """

    def __init__(self, n_inputs, n_features, w_init=None, fix_parameters=False):
        if w_init is None:
            w_init = UniformInitializer((-np.sqrt(3.), np.sqrt(3)))
        w_shape = (n_input, n_features)
        w = nn.Variables.from_numpy_array(
            w_init()).apply(need_grad=not fix_parameters)
        self.W = w

    def __call__(self, inp):
        return F.embed(inp, self.W)
