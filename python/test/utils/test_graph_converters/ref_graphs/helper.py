# Copyright 2018,2019,2020,2021 Sony Corporation.
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

from __future__ import absolute_import

import numpy as np
import nnabla as nn

from nnabla.parameter import get_parameter_or_create


def create_scale_bias(idx, maps, ndim=4, axes=[1]):
    shape = [1] * ndim
    shape[axes[0]] = maps[axes[0]]
    a = get_parameter_or_create("a{}".format(idx), list(shape),
                                None, True, True)
    b = get_parameter_or_create("b{}".format(idx), list(shape),
                                None, True, True)
    return a, b


def get_channel_axes(channel_last=False, dims=2):
    return [dims+1] if channel_last else [1]


def create_conv_weight_bias(inp, maps=16, kernel=(3, 3),
                            channel_last=False, name=''):
    if channel_last:
        channels = inp.shape[-1]
        filter_shape = tuple(kernel) + (channels, )
    else:
        channels = inp.shape[1]
        filter_shape = (channels,) + tuple(kernel)

    w = get_parameter_or_create("w-{}".format(name), (maps,) + filter_shape,
                                None, True, True)
    b = get_parameter_or_create("b-{}".format(name), (maps,),
                                None, True, True)

    return w, b


def create_affine_weight_bias(inp, n_outmaps=10, name=''):
    w = get_parameter_or_create("w-{}".format(name), [int(np.prod(inp.shape[1:]))] + [n_outmaps],
                                None, True, True)
    b = get_parameter_or_create("b-{}".format(name), (n_outmaps, ),
                                None, True, True)

    return w, b
