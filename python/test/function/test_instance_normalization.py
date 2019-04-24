import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F


def ref_instance_normalization(x, beta, gamma, channel_axis, batch_axis, eps, output_stat):

    if hasattr(batch_axis, "__iter__"):
        ignore_axes = batch_axis + [channel_axis, ]
    else:
        ignore_axes = [batch_axis, channel_axis]

    axes = tuple([i for i in range(len(x.shape)) if i not in ignore_axes])

    x_mean = x.mean(axis=axes, keepdims=True)
    x_std = x.std(axis=axes, keepdims=True)

    if output_stat:
        return (x - x_mean) / (x_std + eps) * gamma + beta, x_mean, x_std

    return (x - x_mean) / (x_std + eps) * gamma + beta


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis, channel_axis",
                         [((4, 32, 8, 8), 0, 1),  # convolution (NCHW)
                          ((4, 16, 16, 8), 0, 3),  # convolution (NHWC)
                          ((16, 4), 0, 1),  # affine
                          # time-series (T, B, C) or (B, T, C)
                          ((10, 4, 16), [0, 1], 2)
                          ])
@pytest.mark.parametrize("output_stat", [False, True])
def test_group_normalization_forward_backward(seed, x_shape, batch_axis, channel_axis, output_stat):
    rng = np.random.RandomState(seed)
    input = np.array(rng.randn(*x_shape).astype(np.float32))
    eps = 1e-05

    stat_shape = [1 for _ in range(len(x_shape))]
    if isinstance(batch_axis, int):
        stat_shape[batch_axis] = x_shape[batch_axis]
    else:
        for axis in list(batch_axis):
            stat_shape[axis] = x_shape[axis]
    stat_shape[channel_axis] = x_shape[channel_axis]
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
            assert np.allclose(o.d, r, atol=1e-2, rtol=1e-5)

    else:
        output.forward()
        output.backward()

        assert output.shape == ref.shape
        assert np.allclose(output.d, ref, atol=1e-2, rtol=1e-5)
