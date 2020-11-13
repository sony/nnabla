import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.testing import assert_allclose
from nbla_test_utils import list_context

from nnabla.normalization_functions import _get_axes_excluding

ctxs = list_context('WeightStandardization')


@pytest.fixture(scope="function")
def rng():
    nn.clear_parameters()
    yield np.random.RandomState(313)
    nn.clear_parameters()


def ref_weight_standardization(w, channel_axis, eps, output_stat):
    axes = tuple(_get_axes_excluding(len(w.shape), channel_axis))

    w_mean = w.mean(axis=axes, keepdims=True)
    w_var = w.var(axis=axes, keepdims=True)

    norm = (w - w_mean) / (w_var + eps) ** 0.5

    if output_stat:
        return norm, w_mean, w_var

    return norm


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("w_shape , channel_axis", [((32, 16, 3, 3), 0),  # convolution
                                                    ((8, 4, 16), 2),  # affine
                                                    ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
def test_weight_standardization_forward(rng, ctx, func_name, w_shape, channel_axis, eps, output_stat):
    from nbla_test_utils import function_tester

    w = np.array(rng.randn(*w_shape).astype(np.float32))

    function_tester(rng, F.weight_standardization, ref_weight_standardization, [w], [channel_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=1e-2, backward=[False])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("w_shape , channel_axis", [((8, 4, 3, 3), 0),  # convolution
                                                    ((16, 1), 1),  # affine
                                                    ((4, 2, 8), 2),  # affine
                                                    ])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("output_stat", [False, True])
def test_weight_standardization_forward_backward(rng, ctx, func_name, w_shape, channel_axis, eps, output_stat):
    from nbla_test_utils import function_tester

    w = np.array(rng.randn(*w_shape).astype(np.float32))

    function_tester(rng, F.weight_standardization, ref_weight_standardization, [w], [channel_axis, eps, output_stat], ctx=ctx,
                    func_name=func_name, dstep=1e-2, atol_b=1e-2)


@pytest.mark.parametrize("function , channel_axis, kwargs, param_name",
                         [(PF.convolution, 0, {"outmaps": 10, "kernel": (2, 2)}, "conv/W"),
                          (PF.affine, 1, {"n_outmaps": 10}, "affine/W")])
def test_apply_weight_standardization(rng, function, channel_axis, kwargs, param_name):
    eps = 1e-5
    output_stat = False

    """
    Graph overview of the weight standardization
        inputs    callback           parent      output
        -------   --------         ----------   --------
           x --------------->|
           w ------> WS ---->|----> function -----> y
           b --------------->|
    """

    x = nn.Variable.from_numpy_array(rng.randn(2, 8, 4, 4).astype(np.float32))

    def ws_callback(w): return F.weight_standardization(
        w, channel_axis, eps=eps, output_stat=output_stat)
    y = function(x, apply_w=ws_callback, **kwargs)

    # check forward backward
    y.forward()
    y.backward()

    w = nn.get_parameters()[param_name].d
    w_standardized = y.parent.inputs[1].d

    ref_w_standardized = ref_weight_standardization(
        w, channel_axis, eps, output_stat)

    assert_allclose(w_standardized, ref_w_standardized,
                    atol=1e-02, rtol=1e-5)
