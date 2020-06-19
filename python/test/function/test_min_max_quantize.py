# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

ctxs = list_context('MinMaxQuantize')


def std_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def ref_min_max_quantize(x, qr_min, qr_max, ql_min, ql_max,
                         decay, x_min_max, ema, ste_fine_grained, eps,
                         quantize):
    if not quantize:
        return x

    # min-max
    raxes = tuple([i for i, s in enumerate(ql_min.shape) if s == 1])
    x_min = np.min(x, raxes, keepdims=True)
    x_max = np.max(x, raxes, keepdims=True)
    if x_min_max and ema:
        qr_min = decay * qr_min + (1.0 - decay) * x_min
        qr_max = decay * qr_max + (1.0 - decay) * x_max
    elif x_min_max and not ema:
        qr_min = x_min
        qr_max = x_max
    # scale
    scale = (qr_max - qr_min) / (ql_max - ql_min)
    # nudge scale
    if np.any(qr_max - qr_min < eps):
        qr_max[qr_max - qr_min < eps] = qr_min + eps
    # nudge min-max
    zero_point_from_min = ql_min - qr_min / scale
    zero_point_nudged = std_round(zero_point_from_min)
    if np.any(zero_point_from_min <= ql_min):
        zero_point_nudged[zero_point_from_min <=
                          ql_min] = q_min[zero_point_from_min <= ql_min]
    if np.any(zero_point_from_min >= ql_max):
        zero_point_nudged[zero_point_from_min >=
                          ql_max] = q_max[zero_point_from_min >= ql_max]
    qr_min_nudged = (ql_min - zero_point_nudged) * scale
    qr_max_nudged = (ql_max - zero_point_nudged) * scale
    x_q = std_round((np.clip(x, qr_min_nudged, qr_max_nudged) - qr_min_nudged) /
                    scale) * scale + qr_min_nudged
    return x_q


def ref_grad_min_max_quantize(x, qr_min, qr_max, ql_min, ql_max, dy,
                              decay, x_min_max, ema, ste_fine_grained, eps,
                              quantize, **kw):
    if not quantize:
        return dy.flatten()

    # min-max
    raxes = tuple([i for i, s in enumerate(ql_min.shape) if s == 1])
    x_min = np.min(x, raxes, keepdims=True)
    x_max = np.max(x, raxes, keepdims=True)
    if x_min_max and ema:
        qr_min = decay * qr_min + (1.0 - decay) * x_min
        qr_max = decay * qr_max + (1.0 - decay) * x_max
    elif x_min_max and not ema:
        qr_min = x_min
        qr_max = x_max
    elif not x_min_max and ema:
        pass
    # scale
    scale = (qr_max - qr_min) / (ql_max - ql_min)
    # nudge scale
    if np.any(qr_max - qr_min < eps):
        qr_max[qr_max - qr_min < eps] = qr_min + eps
    # nudge min-max
    zero_point_from_min = ql_min - qr_min / scale
    zero_point_nudged = std_round(zero_point_from_min)
    if np.any(zero_point_from_min <= ql_min):
        zero_point_nudged[zero_point_from_min <=
                          ql_min] = q_min[zero_point_from_min <= ql_min]
    if np.any(zero_point_from_min >= ql_max):
        zero_point_nudged[zero_point_from_min >=
                          ql_max] = q_max[zero_point_from_min >= ql_max]
    qr_min_nudged = (ql_min - zero_point_nudged) * scale
    qr_max_nudged = (ql_max - zero_point_nudged) * scale

    # STE
    dy0 = dy.copy()
    if ste_fine_grained:
        dy = dy0 * (x >= qr_min_nudged) * (x <= qr_max_nudged)
    if x_min_max or ema:
        return dy.flatten()
    else:  # train qr_min and qr_max
        d_qr_min = np.sum(dy0 * (x < qr_min_nudged), raxes, keepdims=True)
        d_qr_max = np.sum(dy0 * (x > qr_max_nudged), raxes, keepdims=True)
        return np.concatenate([dy.flatten(), d_qr_min.flatten(), d_qr_max.flatten()])


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("x_shape, q_shape", [  # per-tensor (per-layer)
                                              ((4, 8, 16, 16), (1, 1, 1, 1)),
                                              # per-channel (channel-first) for activation
                                              ((4, 8, 16, 16), (1, 8, 1, 1)),
                                              # per-channel (channel-first) for weights
                                              ((16, 8, 3, 3), (16, 1, 1, 1)),
                                              ])
@pytest.mark.parametrize("decay", [0.999, 0.9])
@pytest.mark.parametrize("x_min_max", [True, False])
@pytest.mark.parametrize("ema", [True, False])
@pytest.mark.parametrize("ste_fine_grained", [True, False])
@pytest.mark.parametrize("eps", [0.01])
@pytest.mark.parametrize("quantize", [True, False])
def test_min_max_quantize_forward_backward(seed, x_shape, q_shape,
                                           decay, x_min_max, ema, ste_fine_grained,
                                           eps,
                                           quantize,
                                           ctx, func_name):
    from nbla_test_utils import cap_ignore_region, \
        function_tester
    rng = np.random.RandomState(seed)
    # Inputs
    x = rng.randn(*x_shape)
    qr_min = -0.5 * rng.rand(*q_shape)
    qr_max = +0.5 * rng.rand(*q_shape)
    ql_min = np.zeros(q_shape)
    ql_max = np.ones(q_shape) * 255
    inputs = [x, qr_min, qr_max, ql_min, ql_max]
    func_args = [decay, x_min_max, ema, ste_fine_grained, eps, quantize]

    # No quantization
    if not quantize:
        vinputs = [nn.Variable.from_numpy_array(xd) for xd in inputs]
        v = vinputs[0]
        with nn.context_scope(ctx):
            o = F.min_max_quantize(*(vinputs + func_args))
        np.allclose(o.d, v.d)
        return
    # x_min_max and ema
    # function_tester does not work in this combination
    # when comparing gradients between true_g = v.g - g_init (accum=True) and g (accum=False)
    # since forward changes the exponential moving averages
    if x_min_max and ema:
        from nbla_test_utils import ArrayDiffStats
        vinputs = [nn.Variable.from_numpy_array(xd) for xd in inputs]
        vinputs[0].need_grad = True
        with nn.context_scope(ctx):
            y = F.min_max_quantize(*(vinputs + func_args))
        # forward check
        y.forward()
        y_ref = ref_min_max_quantize(x, qr_min, qr_max, ql_min, ql_max,
                                     decay, x_min_max, ema, ste_fine_grained, eps,
                                     quantize)
        assert np.allclose(y_ref, y.d, atol=1e-5), ArrayDiffStats(y_ref, y.d)
        # backward check (accum=False)
        xv = vinputs[0]
        xv.grad.zero()
        dy = rng.randn(*y.shape)
        y.backward(dy)
        gx_ref = ref_grad_min_max_quantize(x, qr_min, qr_max, ql_min, ql_max, dy,
                                           decay, x_min_max, ema, ste_fine_grained, eps,
                                           quantize)
        ag = xv.g.copy()
        assert np.allclose(gx_ref, ag.flatten(),
                           atol=1e-5), ArrayDiffStats(gx_ref, ag.flatten())
        # backward check (accum=True)
        y.backward(dy)
        assert np.allclose(ag * 2.0, xv.g.copy(),
                           atol=1e-5), ArrayDiffStats(ag * 2.0, xv.g.copy())
        return
    # General tests
    backward = [True, False, False, False, False, False] if x_min_max or ema \
        else [True, True, True, False, False, False]
    function_tester(rng,
                    F.min_max_quantize,
                    ref_min_max_quantize,
                    inputs,
                    func_args=func_args,
                    atol_b=1e-3, backward=backward,
                    ctx=ctx, func_name=func_name,
                    disable_half_test=True,
                    ref_grad=ref_grad_min_max_quantize)
