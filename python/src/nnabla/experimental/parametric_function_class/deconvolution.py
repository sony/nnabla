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
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, UniformInitializer)

from .module import Module


class Deconvolution(Module):
    """
    Deconvolution layer.

    Args:
        inp (~nnabla.Variable): N-D array.
        outmaps (int): Number of deconvolution kernels (which is equal to the number of output channels). For example, to apply deconvolution on an input with 16 types of filters, specify 16.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel size. For example, to apply deconvolution on an image with a 3 (height) by 5 (width) two-dimensional kernel, specify (3,5).
        pad (:obj:`tuple` of :obj:`int`): Padding sizes for dimensions.
        stride (:obj:`tuple` of :obj:`int`): Stride sizes for dimensions.
        dilation (:obj:`tuple` of :obj:`int`): Dilation sizes for dimensions.
        group (int): Number of groups of channels. This makes connections across channels sparser by grouping connections along map direction.
        w_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for weight. By default, it is initialized with :obj:`nnabla.initializer.UniformInitializer` within the range determined by :obj:`nnabla.initializer.calc_uniform_lim_glorot`.  
        b_init (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`): Initializer for bias. By default, it is initialized with zeros if `with_bias` is `True`.
        base_axis (int): Dimensions up to `base_axis` are treated as the sample dimensions.
        fix_parameters (bool): When set to `True`, the weights and biases will not be updated.
        rng (numpy.random.RandomState): Random generator for Initializer.
        with_bias (bool): Specify whether to include the bias term.

    Returns:
        :class:`~nnabla.Variable`: N-D array. See :obj:`~nnabla.functions.deconvolution` for the output shape.

    """

    def __init__(self, inmaps, outmaps, kernel,
                 pad=None, stride=None, dilation=None, group=1,
                 w_init=None, b_init=None,
                 base_axis=1, fix_parameters=False, rng=None, with_bias=True):
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(inmaps, outmaps, tuple(kernel)), rng=rng)
        if with_bias and b_init is None:
            b_init = ConstantInitializer()
        w_shape = (outmaps, inmaps // group) + tuple(kernel)
        w = nn.Variable.from_numpy_array(
            w_init(w_shape)).apply(need_grad=not fix_parameters)
        b = None
        if with_bias:
            b_shape = (outmaps, )
            b = nn.Variable.from_numpy_array(
                b_init(b_shape)).apply(need_grad=not fix_parameters)

        self.W = w
        self.b = b
        self.base_axis = base_axis
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group

    def __call__(self, inp):
        return F.deconvolution(inp, self.W, self.b, self.base_axis,
                               self.pad, self.stride, self.dilation, self.group)


Deconv1d = Deconvolution
Deconv2d = Deconvolution
Deconv3d = Deconvolution
DeconvNd = Deconvolution
