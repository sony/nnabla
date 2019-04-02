import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F


def ref_instance_normalization(x, channel_axis, batch_axis, eps, output_stat):

    if hasattr(batch_axis, "__iter__"):
        ignore_axes = batch_axis + [channel_axis, ]
    else:
        ignore_axes = [batch_axis, channel_axis]

    axes = tuple([i for i in range(len(x.shape)) if i not in ignore_axes])

    x_mean = x.mean(axis=axes, keepdims=True)
    x_std = x.std(axis=axes, keepdims=True)

    if output_stat:
        return ((x - x_mean) / (x_std + eps)).reshape(x.shape), x_mean, x_std

    return ((x - x_mean) / (x_std + eps)).reshape(x.shape)


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

    x = nn.Variable.from_numpy_array(input)
    output = F.instance_normalization(
        x, channel_axis, batch_axis, eps, output_stat)
    ref = ref_instance_normalization(
        input, channel_axis, batch_axis, eps, output_stat)

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
