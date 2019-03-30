import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F


def ref_weight_standardization(w, channel_axis, eps, output_stat):
    axes = tuple([i for i in range(len(w.shape)) if i != channel_axis])

    w_mean = w.mean(axis=axes, keepdims=True)
    w_std = w.std(axis=axes, keepdims=True)

    if output_stat:
        return (w - w_mean) / (w_std + eps), w_mean, w_std

    return (w - w_mean) / (w_std + eps)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("w_shape , channel_axis", [((32, 16, 3, 3), 0),  # convolution
                                                    ((16, 1), 1),  # affine
                                                    ((8, 4, 16), 2),  # affine
                                                    ])
@pytest.mark.parametrize("output_stat", [False, True])
def test_weight_standardization_forward_backward(seed, w_shape, channel_axis, output_stat):
    rng = np.random.RandomState(seed)
    input = np.array(rng.randn(*w_shape).astype(np.float32))
    eps = 1e-05

    x = nn.Variable.from_numpy_array(input)
    output = F.weight_standardization(x, channel_axis, eps, output_stat)
    ref = ref_weight_standardization(input, channel_axis, eps, output_stat)

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
