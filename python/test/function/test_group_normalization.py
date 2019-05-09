import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F

from nnabla.normalization import _force_list, _get_axes_excluding


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
    x_std = tmp.std(axis=axes, keepdims=True)

    if output_stat:
        return ((tmp - x_mean) / (x_std + eps) * gamma + beta).reshape(x.shape), x_mean, x_std

    return ((tmp - x_mean) / (x_std + eps) * gamma + beta).reshape(x.shape)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((4, 32, 8, 8), 0, 1),  # convolution (NCHW)
                          ((4, 16, 16, 8), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((10, 4, 16), [0, 1], 2)
                          ])
@pytest.mark.parametrize("output_stat", [False, True])
def test_group_normalization_forward_backward(seed, num_groups, x_shape, batch_axis, channel_axis, output_stat):
    from nnabla.normalization import _force_list

    rng = np.random.RandomState(seed)
    input = np.array(rng.randn(*x_shape).astype(np.float32))

    stat_shape = [x_shape[i] if i in _force_list(
        batch_axis) else 1 for i in range(len(x_shape) + 1)]
    stat_shape[channel_axis] = num_groups
    stat_shape[channel_axis + 1] = int(x_shape[channel_axis] / num_groups)

    beta = rng.randn(*stat_shape).astype(np.float32)
    gamma = rng.randn(*stat_shape).astype(np.float32)

    eps = 1e-05

    x = nn.Variable.from_numpy_array(input)
    v_beta = nn.Variable.from_numpy_array(beta)
    v_gamma = nn.Variable.from_numpy_array(gamma)

    output = F.group_normalization(
        x, v_beta, v_gamma, num_groups, channel_axis, batch_axis, eps, output_stat)
    ref = ref_group_normalization(
        input, beta, gamma, num_groups, channel_axis, batch_axis, eps, output_stat)

    if output_stat:
        tmp = F.sink(*output)
        tmp.forward()
        tmp.backward()

        for o, r in zip(output, ref):
            assert o.shape == r.shape
            assert np.allclose(o.d, r, atol=1e-2, rtol=1e-5)

    else:
        output.forward()
        output.backward()

        assert output.shape == ref.shape
        assert np.allclose(output.d, ref, atol=1e-2, rtol=1e-5)
