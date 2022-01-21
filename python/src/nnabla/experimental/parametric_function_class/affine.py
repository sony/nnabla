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
    calc_uniform_lim_glorot,
    ConstantInitializer, UniformInitializer)

from .module import Module


class Affine(Module):
    """
    The affine layer, also known as the fully connected layer. Computes

    .. math::
        {\\mathbf y} = {\\mathbf A} {\\mathbf x} + {\\mathbf b}.

    where :math:`{\\mathbf x}, {\\mathbf y}` are the inputs and outputs respectively,
    and :math:`{\\mathbf A}, {\\mathbf b}` are constants.

    Args:
        inp (~nnabla.Variable): Input N-D array with shape (:math:`M_0 \\times \ldots \\times M_{B-1} \\times D_B \\times \ldots \\times D_N`). Dimensions before and after base_axis are flattened as if it is a matrix.
        n_outmaps (:obj:`int` or :obj:`tuple` of :obj:`int`): Number of output neurons per data.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: :math:`(B + 1)`-D array. (:math:`M_0 \\times \ldots \\times M_{B-1} \\times L`)f

    """

    def __init__(self, n_inmaps, n_outmaps, base_axis=1, w_init=None, b_init=None,
                 fix_parameters=False, rng=None, with_bias=True):
        if not hasattr(n_outmaps, '__iter__'):
            n_outmaps = [n_outmaps]
        n_outmaps = list(n_outmaps)
        n_outmap = int(np.prod(n_outmaps))
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(n_inmaps, n_outmap), rng=rng)
        if with_bias and b_init is None:
            b_init = ConstantInitializer()
        w_shape = (n_inmaps, n_outmap)
        w = nn.Variable.from_numpy_array(
            w_init(w_shape)).apply(need_grad=not fix_parameters)
        b = None
        if with_bias:
            b_shape = (n_outmap, )
            b = nn.Variable.from_numpy_array(
                b_init(b_shape)).apply(need_grad=not fix_parameters)

        self.W = w
        self.b = b
        self.base_axis = base_axis

    def __call__(self, inp):
        return F.affine(inp, self.W, self.b, self.base_axis)


Linear = Affine
