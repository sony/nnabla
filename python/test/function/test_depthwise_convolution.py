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

from __future__ import division

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import refs
from nbla_test_utils import list_context
from nbla_test_utils import function_tester

ctxs = list_context('DepthwiseConvolution')


def ref_depthwise_convolution_1d(x, w, b, base_axis, pad, stride, dilation,
                                 multiplier):
    """ implement depthwise convolution by using normal convolution
    with group = inmaps """
    # insert second dimension to weights
    w = np.expand_dims(w, axis=1)
    y = []

    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        groups = xx.shape[0]
        yy = refs.convolution_1d(xx, w, b, pad, stride, dilation, groups)
        y.append(yy[np.newaxis])

    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


def ref_depthwise_convolution_2d(x, w, b, base_axis, pad, stride, dilation,
                                 multiplier):
    """ implement depthwise convolution by using normal convolution
    with group = inmaps """
    # insert second dimension to weights
    w = np.expand_dims(w, axis=1)
    y = []

    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        groups = xx.shape[0]
        yy = refs.convolution_2d(xx, w, b, pad, stride, dilation, groups)
        y.append(yy[np.newaxis])

    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, pad, stride, dilation", [
    ((2, 4, 10, 10), (3, 2), (0, 0), (1, 1), (1, 1)),
    ((2, 2, 10, 10), (3, 3), (3, 0), (1, 2), (2, 1)),
])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("multiplier", [1, 3])
def test_forward_backward_2d(inshape, kernel, pad, stride, dilation, with_bias,
                             multiplier, seed, ctx, func_name):
    base_axis = len(inshape) - 3
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels * multiplier
    rng = np.random.RandomState(seed)
    x = rng.randn(*inshape).astype(np.float32)
    w = rng.randn(*((outmap_channels,) + kernel)).astype(np.float32)
    b = rng.randn(outmap_channels).astype(np.float32) if with_bias else None
    inputs = [x, w, b]
    func_args = [base_axis, pad, stride, dilation, multiplier]
    reference = ref_depthwise_convolution_2d
    function_tester(rng, F.depthwise_convolution, reference, inputs,
                    func_args, atol_f=1e-4, atol_b=3e-3, dstep=1e-2,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, pad, stride, dilation", [
    ((2, 2, 10), (3,), (3,), (1,), (2,)),
    ((2, 4, 10), (3,), (0,), (2,), (1,)),
    ((2, 2, 10), (4,), (3,), (1,), (2,)),
])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("multiplier", [1, 3])
def test_forward_backward_1d(inshape, kernel, pad, stride, dilation, with_bias,
                             multiplier, seed, ctx, func_name):
    base_axis = len(inshape) - 2
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels * multiplier
    rng = np.random.RandomState(seed)
    x = rng.randn(*inshape).astype(np.float32)
    w = rng.randn(*((outmap_channels,) + kernel)).astype(np.float32)
    b = rng.randn(outmap_channels).astype(np.float32) if with_bias else None
    inputs = [x, w, b]
    func_args = [base_axis, pad, stride, dilation, multiplier]
    reference = ref_depthwise_convolution_1d
    function_tester(rng, F.depthwise_convolution, reference, inputs,
                    func_args, atol_f=1e-4, atol_b=3e-3, dstep=1e-2,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("inshape, kernel, multiplier, outshape", [
    ((2, 2, 10), (3,), 1, (2, 2, 8)),
    ((2, 2, 10), (3,), 3, (2, 6, 8)),
])
def test_parametric_function_1d(inshape, kernel, multiplier, outshape):
    base_axis = len(inshape) - 2
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels * multiplier
    x = nn.Variable(inshape)
    y = PF.depthwise_convolution(x, kernel, multiplier=multiplier)
    p = nn.get_parameters()
    assert y.shape == outshape
    assert p['depthwise_conv/W'].shape == (outmap_channels,) + kernel
    assert p['depthwise_conv/b'].shape == (outmap_channels,)
    nn.clear_parameters()


@pytest.mark.parametrize("inshape, kernel, multiplier, outshape", [
    ((2, 2, 10, 10), (3, 2), 1, (2, 2, 8, 9)),
    ((2, 2, 10, 10), (3, 2), 2, (2, 4, 8, 9)),
])
def test_parametric_function_2d(inshape, kernel, multiplier, outshape):
    base_axis = len(inshape) - 3
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels * multiplier
    x = nn.Variable(inshape)
    y = PF.depthwise_convolution(x, kernel, multiplier=multiplier)
    p = nn.get_parameters()
    assert y.shape == outshape
    assert p['depthwise_conv/W'].shape == (outmap_channels,) + kernel
    assert p['depthwise_conv/b'].shape == (outmap_channels,)
    nn.clear_parameters()
