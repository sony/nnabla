import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.testing import assert_allclose

from nnabla.normalization_functions import _force_list, _get_axes_excluding


def ref_layer_normalization(x, beta, gamma, batch_axis, eps, output_stat):
    batch_axis = _force_list(batch_axis)

    axes = tuple(_get_axes_excluding(len(x.shape), batch_axis))

    x_mean = x.mean(axis=axes, keepdims=True)
    x_var = x.var(axis=axes, keepdims=True)

    norm = (x - x_mean) / (x_var + eps) ** 0.5

    if output_stat:
        return norm * gamma + beta, x_mean, x_var

    return norm * gamma + beta


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

    stat_shape = list(x_shape)
    for baxis in _force_list(batch_axis):
        stat_shape[baxis] = 1

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
            assert_allclose(o.d, r, atol=1e-2, rtol=1e-5)

    else:
        output.forward()
        output.backward()

        assert_allclose(output.d, ref, atol=1e-2, rtol=1e-5)
