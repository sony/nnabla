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

ctxs = list_context('DepthwiseDeconvolution')


def ref_depthwise_deconvolution_1d(x, w, b, base_axis, pad, stride, dilation,
                                   divisor):
    """ implement depthwise deconvolution by using normal deconvolution
    with group = inmaps / divisor """
    # insert second dimension to weights
    w = np.expand_dims(w, axis=1)
    y = []

    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        groups = xx.shape[0] // divisor
        yy = refs.deconvolution_1d(xx, w, b, pad, stride, dilation, groups)
        y.append(yy[np.newaxis])

    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


def ref_depthwise_deconvolution_2d(x, w, b, base_axis, pad, stride, dilation,
                                   divisor):
    """ implement depthwise deconvolution by using normal deconvolution
    with group = inmaps """
    # insert second dimension to weights
    w = np.expand_dims(w, axis=1)
    y = []

    for xx in x.reshape((-1,) + x.shape[base_axis:]):
        groups = xx.shape[0] // divisor
        yy = refs.deconvolution_2d(xx, w, b, pad, stride, dilation, groups)
        y.append(yy[np.newaxis])

    y = np.vstack(y)
    return y.reshape(x.shape[:base_axis] + y.shape[1:])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, pad, stride, dilation, divisor", [
    ((2, 2, 10, 10), (3, 2), (3, 0), (1, 2), (2, 1), 1),
    ((2, 3, 10, 10), (3, 2), (3, 0), (1, 2), (2, 1), 3),
    ((2, 4, 10, 10), (3, 2), (0, 0), (1, 1), (1, 1), 1),
    ((2, 6, 10, 10), (3, 2), (0, 0), (1, 1), (1, 1), 2),
    ((3, 2, 10, 10), (3, 3), (3, 0), (1, 2), (2, 1), 1),
    ((3, 2, 10, 10), (3, 3), (3, 0), (1, 2), (2, 1), 2),
])
@pytest.mark.parametrize("with_bias", [True, False])
def test_depthwise_deconvolution_2d_forward_backward(inshape, kernel, pad,
                                                     stride, dilation, divisor,
                                                     with_bias, seed, ctx,
                                                     func_name):
    base_axis = len(inshape) - 3
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels // divisor
    rng = np.random.RandomState(seed)
    x = rng.randn(*inshape).astype(np.float32)
    w = rng.randn(*((sample_channels,) + kernel)).astype(np.float32)
    b = rng.randn(outmap_channels).astype(np.float32) if with_bias else None
    inputs = [x, w, b]
    func_args = [base_axis, pad, stride, dilation, divisor]
    reference = ref_depthwise_deconvolution_2d
    function_tester(rng, F.depthwise_deconvolution, reference, inputs,
                    func_args, atol_f=1e-4, atol_b=4e-3, dstep=1e-2,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, kernel, pad, stride, dilation, divisor", [
    ((2, 2, 10), (3,), (3,), (1,), (2,), 1),
    ((2, 3, 10), (3,), (3,), (1,), (2,), 3),
    ((2, 4, 10), (3,), (0,), (2,), (1,), 1),
    ((2, 4, 10), (3,), (0,), (2,), (1,), 2),
    ((3, 2, 10), (4,), (3,), (1,), (2,), 1),
    ((3, 9, 10), (4,), (3,), (1,), (2,), 3),
])
@pytest.mark.parametrize("with_bias", [True, False])
def test_depthwise_deconvolution_1d_forward_backward(inshape, kernel, pad,
                                                     stride, dilation, divisor,
                                                     with_bias, seed, ctx,
                                                     func_name):
    base_axis = len(inshape) - 2
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels // divisor
    rng = np.random.RandomState(seed)
    x = rng.randn(*inshape).astype(np.float32)
    w = rng.randn(*((sample_channels,) + kernel)).astype(np.float32)
    b = rng.randn(outmap_channels).astype(np.float32) if with_bias else None
    inputs = [x, w, b]
    func_args = [base_axis, pad, stride, dilation, divisor]
    reference = ref_depthwise_deconvolution_1d
    function_tester(rng, F.depthwise_deconvolution, reference, inputs,
                    func_args, atol_f=1e-4, atol_b=3e-3, dstep=1e-2,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("inshape, kernel, divisor, outshape", [
    ((2, 4, 10), (3,), 1, (2, 4, 12)),
    ((2, 4, 10), (3,), 2, (2, 2, 12)),
])
def test_parametric_function_1d(inshape, kernel, divisor, outshape):
    base_axis = len(inshape) - 2
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels // divisor
    x = nn.Variable(inshape)
    y = PF.depthwise_deconvolution(x, kernel, divisor=divisor)
    p = nn.get_parameters()
    assert y.shape == outshape
    assert p['depthwise_deconv/W'].shape == (sample_channels,) + kernel
    assert p['depthwise_deconv/b'].shape == (outmap_channels,)
    nn.clear_parameters()


@pytest.mark.parametrize("inshape, kernel, divisor, outshape", [
    ((2, 4, 10, 10), (3, 2), 1, (2, 4, 12, 11)),
    ((2, 4, 10, 10), (3, 2), 2, (2, 2, 12, 11)),
])
def test_parametric_function_2d(inshape, kernel, divisor, outshape):
    base_axis = len(inshape) - 3
    sample_channels = inshape[base_axis]
    outmap_channels = sample_channels // divisor
    x = nn.Variable(inshape)
    y = PF.depthwise_deconvolution(x, kernel, divisor=divisor)
    p = nn.get_parameters()
    assert y.shape == outshape
    assert p['depthwise_deconv/W'].shape == (sample_channels,) + kernel
    assert p['depthwise_deconv/b'].shape == (outmap_channels,)
    nn.clear_parameters()
