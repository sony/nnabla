import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context

from nnabla.normalization_functions import _force_list, _get_axes_excluding

ctxs = list_context('GroupNormalization')


def ref_group_normalization(x, beta, gamma, num_groups, channel_axis, batch_axis, eps, output_stat):
    cdim = x.shape[channel_axis]

    if cdim % num_groups > 0:
        raise ValueError()

    shape = x.shape[:channel_axis] + (num_groups, int(cdim / num_groups))
    if channel_axis < len(x.shape) - 1:
        shape += x.shape[channel_axis + 1:]

    tmp = x.reshape(shape).copy()

    ignore_axes = _force_list(batch_axis) + [channel_axis, ]

    axes = tuple(_get_axes_excluding(len(shape), ignore_axes))

    x_mean = tmp.mean(axis=axes, keepdims=True)
    x_var = tmp.var(axis=axes, keepdims=True)

    norm = (tmp - x_mean) / (x_var + eps) ** 0.5

    norm = norm.reshape(x.shape)

    if gamma is not None:
        norm *= gamma

    if beta is not None:
        norm += beta

    if output_stat:
        return norm, x_mean, x_var

    return norm


def create_inputs(rng, x_shape, channel_axis, no_scale, no_bias):
    x = np.array(rng.randn(*x_shape).astype(np.float32))

    stat_shape = [1 for _ in range(len(x_shape))]
    stat_shape[channel_axis] = x_shape[channel_axis]

    beta = None if no_bias else rng.randn(*stat_shape).astype(np.float32)
    gamma = None if no_scale else rng.randn(*stat_shape).astype(np.float32)

    return x, beta, gamma


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("num_groups", [2, 3])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((4, 24, 8, 8), 0, 1),  # convolution (NCHW)
                          ((4, 16, 16, 6), 0, 3),  # convolution (NHWC)
                          ((8, 6), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((4, 3, 6), [0, 1], 2)
                          ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_group_normalization_forward(ctx, func_name, seed, num_groups, x_shape, batch_axis, channel_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(
        rng, x_shape, channel_axis, no_scale, no_bias)

    function_tester(rng, F.group_normalization, ref_group_normalization, [x, beta, gamma], [num_groups, channel_axis, batch_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=4e-2, atol_accum=1e-5, backward=[False, False, False], disable_half_test=True)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("num_groups", [2, 3])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((2, 6, 3, 3), 0, 1),  # convolution (NCHW)
                          ((2, 3, 3, 6), 0, 3),  # convolution (NHWC)
                          ((8, 6), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((4, 3, 6), [0, 1], 2)
                          ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
@pytest.mark.parametrize("no_scale", [False, True])
@pytest.mark.parametrize("no_bias", [False, True])
def test_group_normalization_forward_backward(ctx, func_name, seed, num_groups, x_shape, batch_axis, channel_axis, eps, output_stat, no_scale, no_bias):
    from nbla_test_utils import function_tester

    rng = np.random.RandomState(seed)
    x, beta, gamma = create_inputs(
        rng, x_shape, channel_axis, no_scale, no_bias)

    function_tester(rng, F.group_normalization, ref_group_normalization, [x, beta, gamma], [num_groups, channel_axis, batch_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=4e-2, atol_accum=1e-5, backward=[True, not no_bias, not no_scale], disable_half_test=True)
