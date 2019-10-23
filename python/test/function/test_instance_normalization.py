import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.testing import assert_allclose

from nnabla.normalization_functions import _force_list, _get_axes_excluding


def ref_instance_normalization(x, beta, gamma, channel_axis, batch_axis, eps, output_stat):

    ignore_axes = _force_list(batch_axis) + [channel_axis, ]

    axes = tuple(_get_axes_excluding(len(x.shape), ignore_axes))

    x_mean = x.mean(axis=axes, keepdims=True)
    x_var = x.var(axis=axes, keepdims=True)

    norm = (x - x_mean) / (x_var + eps) ** 0.5

    if output_stat:
        return norm * gamma + beta, x_mean, x_var

    return norm * gamma + beta


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((4, 32, 8, 8), 0, 1),  # convolution (NCHW)
                          ((4, 16, 16, 8), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((10, 4, 16), [0, 1], 2)
                          ])
@pytest.mark.parametrize("output_stat", [False, True])
def test_instance_normalization_forward_backward(seed, x_shape, batch_axis, channel_axis, output_stat):

    rng = np.random.RandomState(seed)
    input = np.array(rng.randn(*x_shape).astype(np.float32))
    eps = 1e-05

    stat_shape = tuple([x_shape[i] if i in _force_list(batch_axis) + [channel_axis, ] else 1
                        for i in range(len(x_shape))])

    beta = rng.randn(*stat_shape).astype(np.float32)
    gamma = rng.randn(*stat_shape).astype(np.float32)

    x = nn.Variable.from_numpy_array(input)
    v_beta = nn.Variable.from_numpy_array(beta)
    v_gamma = nn.Variable.from_numpy_array(gamma)

    output = F.instance_normalization(
        x, v_beta, v_gamma, channel_axis, batch_axis, eps, output_stat)
    ref = ref_instance_normalization(
        input, beta, gamma, channel_axis, batch_axis, eps, output_stat)

    if output_stat:
        tmp = F.sink(*output)
        tmp.forward()
        tmp.backward()

        for o, r in zip(output, ref):
            assert o.shape == r.shape
            assert_allclose(o.d, r, atol=1e-2, rtol=1e-5)

    else:
        output.forward()
        output.backward()

        assert output.shape == ref.shape
        assert_allclose(output.d, ref, atol=1e-2, rtol=1e-5)
