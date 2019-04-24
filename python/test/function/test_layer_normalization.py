import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F


def ref_layer_normalization(x, beta, gamma, batch_axis, eps, output_stat):
    if not hasattr(batch_axis, "__iter__"):
        batch_axis = [batch_axis]

    axes = tuple([i for i in range(len(x.shape)) if i not in batch_axis])

    x_mean = x.mean(axis=axes, keepdims=True)
    x_std = x.std(axis=axes, keepdims=True)

    if output_stat:
        return (x - x_mean) / (x_std + eps) * gamma + beta, x_mean, x_std

    return (x - x_mean) / (x_std + eps) * gamma + beta


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape , batch_axis", [((4, 3, 8, 8), 0),
                                                  ((4, 16, 16, 8), 0),
                                                  ((16, 1), 0),
                                                  ((3, 32, 4), 0),
                                                  # time-series (T, B, C) or (B, T, C)
                                                  ((10, 4, 16), [0, 1])
                                                  ])
@pytest.mark.parametrize("output_stat", [False, True])
def test_layer_normalization_forward_backward(seed, x_shape, batch_axis, output_stat):
    rng = np.random.RandomState(seed)
    input = rng.randn(*x_shape).astype(np.float32)

    stat_shape = [1 for _ in range(len(x_shape))]
    if isinstance(batch_axis, int):
        stat_shape[batch_axis] = x_shape[batch_axis]
    else:
        for axis in list(batch_axis):
            stat_shape[axis] = x_shape[axis]
    beta = rng.randn(*stat_shape).astype(np.float32)
    gamma = rng.randn(*stat_shape).astype(np.float32)
    eps = 1e-05

    x = nn.Variable.from_numpy_array(input)
    v_beta = nn.Variable.from_numpy_array(beta)
    v_gamma = nn.Variable.from_numpy_array(gamma)

    output = F.layer_normalization(
        x, v_beta, v_gamma, batch_axis, eps, output_stat)
    ref = ref_layer_normalization(
        input, beta, gamma, batch_axis, eps, output_stat)

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

        assert np.allclose(output.d, ref, atol=1e-2, rtol=1e-5)
