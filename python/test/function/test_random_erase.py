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
from test_flip import ref_flip
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('RandomErase')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("prob", [0.7, 1.0])
@pytest.mark.parametrize("area_ratios", [(0.02, 0.04)])
@pytest.mark.parametrize("aspect_ratios", [(0.3, 3.3333)])
@pytest.mark.parametrize("replacements", [(2.0, 2.0), (3.0, 4.0)])
@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("inplace", [False])
@pytest.mark.parametrize("base_axis", [1])
@pytest.mark.parametrize("func_seed", [412, -1])
@pytest.mark.parametrize("channel_last", [False, True])
def test_random_erase_forward(ctx, func_name, seed, prob,
                              area_ratios, aspect_ratios, replacements,
                              n, share, inplace, base_axis, func_seed, channel_last):
    if channel_last and func_name == "RandomErase":
        pytest.skip(
            "RandomErase with channel_last is only supported in CUDA.")

    lb = replacements[0]
    rng = np.random.RandomState(seed)
    b, c, h, w = 4, 3, 32, 32
    ishape = [b, h, w, c] if channel_last else [b, c, h, w]
    x = nn.Variable.from_numpy_array(
        rng.rand(*ishape) + 1.0)
    with nn.context_scope(ctx):
        y0 = F.random_erase(x, prob=prob,
                            area_ratios=area_ratios,
                            aspect_ratios=aspect_ratios,
                            replacements=replacements,
                            n=n, share=share, inplace=inplace, base_axis=base_axis,
                            seed=func_seed, channel_last=channel_last)
    # Deterministic check
    y0.forward()
    if prob == 1.0:
        assert np.any(y0.d >= lb)

    # Random but with same seed check
    if func_seed != -1:
        with nn.context_scope(ctx):
            y1 = F.random_erase(x, prob=prob,
                                area_ratios=area_ratios,
                                aspect_ratios=aspect_ratios,
                                replacements=replacements,
                                n=n, share=share, inplace=inplace, base_axis=base_axis,
                                seed=func_seed, channel_last=channel_last)
        y1.forward()
        assert_allclose(y0.d, y1.d)

    # Random but with different seed check
    with nn.context_scope(ctx):
        y2 = F.random_erase(x, prob=prob,
                            area_ratios=area_ratios,
                            aspect_ratios=aspect_ratios,
                            replacements=replacements,
                            n=n, share=share, inplace=inplace, base_axis=base_axis,
                            seed=func_seed + 2, channel_last=channel_last)
    y2.forward()
    assert np.any(y0.d != y2.d)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("prob", [0.7, 1.0])
@pytest.mark.parametrize("area_ratios", [(0.02, 0.04)])
@pytest.mark.parametrize("aspect_ratios", [(0.3, 3.3333)])
@pytest.mark.parametrize("replacements", [(2.0, 2.0), (3.0, 4.0)])
@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("inplace", [False])
@pytest.mark.parametrize("base_axis", [1])
@pytest.mark.parametrize("func_seed", [412, -1])
@pytest.mark.parametrize("channel_last", [False, True])
@pytest.mark.parametrize("ste_fine_grained", [True, False])
def test_random_erase_backward(ctx, func_name, seed, prob,
                               area_ratios, aspect_ratios, replacements,
                               n, share, inplace, base_axis, func_seed, channel_last, ste_fine_grained):
    if channel_last and func_name == "RandomErase":
        pytest.skip(
            "RandomErase with channel_last is only supported in CUDA.")

    lb = replacements[0]
    rng = np.random.RandomState(seed)
    b, c, h, w = 4, 3, 32, 32
    ishape = [b, c, h, w]

    # Backward check for accum
    x = nn.Variable.from_numpy_array(
        rng.rand(*ishape) + 1.0).apply(need_grad=True)
    xg = rng.randn(*x.shape)
    x.g = xg
    with nn.context_scope(ctx):
        y = F.random_erase(x, prob=prob,
                           area_ratios=area_ratios,
                           aspect_ratios=aspect_ratios,
                           replacements=replacements,
                           n=n, share=share, inplace=inplace, base_axis=base_axis,
                           seed=func_seed, channel_last=channel_last,
                           ste_fine_grained=ste_fine_grained)
    y.forward()
    y.backward(clear_buffer=True)
    if ste_fine_grained:
        xg[np.where(y.d < lb)] += 1.0
        assert_allclose(x.g, xg)
    else:
        assert_allclose(x.g, xg + 1.0)

    # Backward check for not accum
    x = nn.Variable.from_numpy_array(
        rng.rand(*ishape) + 1.0).apply(need_grad=True)
    y = F.identity(x)
    with nn.context_scope(ctx):
        z = F.random_erase(y, prob=prob,
                           area_ratios=area_ratios,
                           aspect_ratios=aspect_ratios,
                           replacements=replacements,
                           n=n, share=share, inplace=inplace, base_axis=base_axis,
                           seed=func_seed, channel_last=channel_last,
                           ste_fine_grained=ste_fine_grained)
    z.forward()
    z.backward(clear_buffer=False)
    if ste_fine_grained:
        assert_allclose(y.g[np.where(z.d >= lb)], 0.0)
        assert_allclose(y.g[np.where(z.d < lb)], 1.0)
    else:
        assert_allclose(y.g, 1.0)
