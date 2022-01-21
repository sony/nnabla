# Copyright 2021 Sony Group Corporation.
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
from nbla_test_utils import list_context

ctxs = list_context('BoolScatter')


def ref_bool_scatter(sdata, mask):
    gdata_shape = mask.shape + sdata.shape[1:]
    mask_bool = mask.astype(np.bool)
    gdata = np.zeros(gdata_shape)
    gdata[mask_bool] = sdata
    return gdata


def ref_bool_scatter_inplace(sdata, mask, gdata):
    mask_bool = mask.astype(np.bool)
    gdata0 = np.copy(gdata)
    gdata0[mask_bool] = sdata
    return gdata0


@pytest.mark.parametrize("gshape, mask_shape",
                         [((2, 3, 2), (2, 3)),
                          ((3, 4), (3, 4)),
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_scatter_forward_backward(seed, ctx, func_name, gshape, mask_shape):
    from nbla_test_utils import cap_ignore_region, function_tester

    rng = np.random.RandomState(seed)
    gdata0 = rng.randn(*gshape).astype(np.float32)
    mask = rng.randint(0, 2, size=mask_shape)
    sdata = gdata0[mask.astype(np.bool)]

    inputs = [sdata, mask]
    backward = [True, False]

    function_tester(rng, F.bool_scatter, ref_bool_scatter, inputs,
                    ctx=ctx, func_name=func_name, func_args=[],
                    backward=backward)


@pytest.mark.parametrize("gshape, mask_shape",
                         [((2, 3, 2), (2, 3)),
                          ((3, 4), (3, 4)),
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_scatter_double_backward(seed, ctx, func_name, gshape, mask_shape):
    from nbla_test_utils import backward_function_tester

    rng = np.random.RandomState(seed)
    gdata0 = rng.randn(*gshape).astype(np.float32)
    mask = rng.randint(0, 2, size=mask_shape).astype(np.float32)
    sdata = gdata0[mask.astype(np.bool)]
    inputs = [sdata, mask]
    backward = [True, False]

    backward_function_tester(rng, F.bool_scatter, inputs, ctx=ctx,
                             backward=[True, False],
                             backward_b=[True, True, False],
                             auto_forward=True)


@pytest.mark.parametrize("gshape, mask_shape",
                         [((2, 3, 2), (2, 3)),
                          ((3, 4), (3, 4)),
                          ])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
def test_bool_scatter_inplace(seed, ctx, func_name, gshape, mask_shape):
    from nbla_test_utils import inplace_function_test_helper

    rng = np.random.RandomState(seed)
    gdata0 = rng.randn(*gshape).astype(np.float32)
    mask = rng.randint(0, 2, size=mask_shape)
    sdata = gdata0[mask.astype(np.bool)]
    gdata1 = rng.randn(*gshape).astype(np.float32)

    v_sdata = nn.Variable.from_numpy_array(sdata).apply(need_grad=True)
    v_mask = nn.Variable.from_numpy_array(mask)
    v_gdata1 = nn.Variable.from_numpy_array(gdata1).apply(need_grad=True)

    with nn.auto_forward():
        v_gdata2 = F.bool_scatter(v_sdata, v_mask, v_gdata1)

    # inplace check
    np.testing.assert_allclose(v_gdata2.d, v_gdata1.d,
                               err_msg="F.bool_scatter(inplace) is not inplaced.")

    # ref check
    gdata2 = ref_bool_scatter_inplace(sdata, mask, gdata1)
    np.testing.assert_allclose(v_gdata2.d, gdata2,
                               err_msg="F.bool_scatter(inplace) fails.")

    # backward wrt inplaced variable (wrt sdata is checked in not-inplaced case)
    egrad = rng.randn(*gdata2.shape)
    mask = mask if mask.shape == gdata1.shape else \
        mask.reshape(mask.shape + (1, ) * (gdata1.ndim - mask.ndim))
    ref_grad = egrad * (1 - mask)
    v_gdata1.grad.fill(0)
    v_gdata2.backward(egrad)
    np.testing.assert_allclose(v_gdata1.g, ref_grad,
                               err_msg="F.bool_scatter(inplace) backward wrt inplace data fails.")

    bgrad = rng.randn(*gdata1.shape)
    v_gdata1.g = bgrad
    v_gdata2.backward(egrad)
    np.testing.assert_allclose(v_gdata1.g - bgrad, ref_grad, atol=1e-6,
                               err_msg="F.bool_scatter(inplace) backward (accum) wrt inplace data fails.")

    # nn.grad (wrt sdata is checked in not-inplaced case)
    with nn.auto_forward():
        d_gdata1 = nn.grad([v_gdata2], [v_gdata1], grad_outputs=[egrad])
    np.testing.assert_allclose(d_gdata1[0].d, ref_grad, atol=1e-6,
                               err_msg="nn.grad (F.bool_scatter(inplace)) wrt inplace data fails.")
