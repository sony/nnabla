# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.testing import assert_allclose


from nbla_test_utils import (function_tester, list_context,
                             list_ctx_and_func_name)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, (0, 2), (1, 2, 3)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("inshape", [(2, 3, 4, 5), (2, 1, 4, 5)])
@pytest.mark.parametrize("op, ctx, func_name", list_ctx_and_func_name(['sum', 'mean', 'max', 'min', 'prod']))
def test_reduction_forward_backward(op, seed, inshape, axis, keepdims, ctx, func_name):
    func = getattr(F, op)
    ref_func = getattr(np, op)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, func, ref_func, inputs,
                    func_args=[axis],
                    func_kwargs=dict(keepdims=keepdims),
                    ctx=ctx, func_name=func_name,
                    # The backward test on macOS doesn't pass with this tolerance.
                    # Does Eigen library used in CPU computation backend produce
                    # the different results on different platforms?
                    # atol_b=3e-3,
                    atol_b=6e-3)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, axis", [((2, 1, 64, 64), (2, 3))])
@pytest.mark.parametrize("ctx, func_name", list_context('Sum'))
def test_large_sum_reduction(seed, inshape, axis, ctx, func_name):
    if not func_name.endswith('Cuda'):
        # This configuration is only to test the CUDA implementation branch
        # where reduction_size / outer_size >= 2048, so we skip this test for
        # CPU and CuDNN and also do not need to run for both keepdims.
        pytest.skip('skip CUDA specific implementation test for other target')

    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.sum, np.sum, inputs, ctx=ctx, func_name=func_name,
                    func_args=[axis], atol_b=1e-2, disable_half_test=False)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape, axis", [((2, 1, 64, 64), (2, 3))])
@pytest.mark.parametrize("ctx, func_name", list_context('Mean'))
def test_large_mean_reduction(seed, inshape, axis, ctx, func_name):
    if not func_name.endswith('Cuda'):
        # This configuration is only to test the CUDA implementation branch
        # where reduction_size / outer_size >= 2048, so we skip this test for
        # CPU and CuDNN and also do not need to run for both keepdims.
        pytest.skip('skip CUDA specific implementation test for other target')

    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.mean, np.mean, inputs, ctx=ctx, func_name=func_name,
                    func_args=[axis])


def preprocess_arg_reduction(x, axes):
    r"""
    Transpose and reshape the input numpy.ndarray x for 2D shape
    (outer, reduction_axes). This is required because numpy cannot calculate
    argmmin/argmax for multi-dimension.

    Args:
        x (numpy.ndarray): Input array.
        axis (:obj:`tuple` of :obj:`int`): Argmax/argmin along these axes.
    """
    shape = np.array(x.shape)
    outer_axes = list(np.delete(np.arange(x.ndim), axes))
    outer_size = np.prod(shape[outer_axes])
    reduction_axes = list(axes)
    reduction_size = np.prod(shape[reduction_axes])
    transpose_axes = outer_axes + reduction_axes
    argmax_shape = [outer_size, reduction_size]
    return x.transpose(transpose_axes).reshape(*argmax_shape)


def argmax(x, axes):
    """
    Args:
        x (numpy.ndarray): Input array.
        axis (:obj:`tuple` of :obj:`int`): Argmax along these axes.
    """
    return preprocess_arg_reduction(x, axes).argmax(-1)


def argmin(x, axis):
    """
    Args:
        x (numpy.ndarray): Input array.
        axis (:obj:`tuple` of :obj:`int`): Argmin along these axes.
    """
    return preprocess_arg_reduction(x, axis).argmin(-1)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ctx, func_name", list_context('Max'))
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("inshape, axis", [
    ((8, 1, 128, 128), (2, 3)),
    ((8, 128, 1, 128), (1, 3)),
    ((8, 128, 8, 128), (1, 3)),
])
def test_max_with_index(seed, ctx, func_name, inshape, axis, keepdims):
    x = np.random.RandomState(seed).randn(*inshape).astype(np.float32)
    x = nn.Variable.from_numpy_array(x)

    # with_index
    with nn.context_scope(ctx), nn.auto_forward(True):
        val, idx = F.max(x, axis, keepdims, with_index=True)
    assert_allclose(val.d, np.amax(x.d, axis, keepdims=keepdims))
    assert np.all(idx.d == argmax(x.d, axis).reshape(idx.d.shape))

    # only_index
    with nn.context_scope(ctx), nn.auto_forward(True):
        idx = F.max(x, axis, keepdims, only_index=True)
    assert np.all(idx.d == argmax(x.d, axis).reshape(idx.d.shape))


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("ctx, func_name", list_context('Min'))
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("inshape, axis", [
    ((8, 1, 128, 128), (2, 3)),
    ((8, 128, 1, 128), (1, 3)),
    ((8, 128, 8, 128), (1, 3)),
])
def test_min_with_index(seed, ctx, func_name, inshape, axis, keepdims):
    x = np.random.RandomState(seed).randn(*inshape).astype(np.float32)
    x = nn.Variable.from_numpy_array(x)

    # with_index
    with nn.context_scope(ctx), nn.auto_forward(True):
        val, idx = F.min(x, axis, keepdims, with_index=True)
    assert_allclose(val.d, np.amin(x.d, axis, keepdims=keepdims))
    assert np.all(idx.d == argmin(x.d, axis).reshape(idx.d.shape))

    # only_index
    with nn.context_scope(ctx), nn.auto_forward(True):
        idx = F.min(x, axis, keepdims, only_index=True)
    assert np.all(idx.d == argmin(x.d, axis).reshape(idx.d.shape))


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3, (0, 2), (1, 2, 3)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("inshape", [(2, 3, 4, 5), (2, 1, 4, 5)])
@pytest.mark.parametrize("op, ctx, func_name", list_ctx_and_func_name(['sum', 'mean', 'max', 'min', 'prod']))
def test_reduction_double_backward(op, seed, inshape, axis, keepdims, ctx, func_name):
    from nbla_test_utils import backward_function_tester
    func = getattr(F, op)
    ref_func = getattr(np, op)
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    backward_function_tester(rng, func, inputs,
                             func_args=[axis],
                             func_kwargs=dict(keepdims=keepdims),
                             ctx=ctx,
                             atol_accum=8e-2)
