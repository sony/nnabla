# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('PatchCorrelation')


@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("C", [1, 4])
@pytest.mark.parametrize("H, W, params, oCH, oCW, oH, oW", [
    (5, 5, {}, 1, 1, 5, 5),
    (5, 5, {'patch': 3}, 1, 1, 3, 3),
    (5, 5, {'shift': 2}, 5, 5, 5, 5),
    (5, 5, {'shift': 2, 'shift_step': 2}, 3, 3, 5, 5),
    (5, 5, {'patch': 3, 'padding': 2}, 1, 1, 7, 7),
    (5, 5, {'patch': 3, 'patch_step': 2, 'padding': 2}, 1, 1, 4, 4),
    (5, 5, {'patch': 4, 'patch_step': 2, 'padding': 2}, 1, 1, 3, 3),
    (7, 8, {'patch': 4, 'patch_step': 1}, 1, 1, 4, 5),
    (7, 8, {'patch': 4, 'patch_step': 2}, 1, 1, 2, 3),
    (7, 8, {'patch': 4, 'patch_step': 3}, 1, 1, 2, 2),
    (3, 3, {'patch': 3, 'padding': 0}, 1, 1, 1, 1),
    (3, 3, {'patch': 3, 'padding': 1}, 1, 1, 3, 3),
    (3, 3, {'patch': 2, 'padding': 0}, 1, 1, 2, 2),
    (3, 3, {'patch': 2, 'padding': 1}, 1, 1, 4, 4),
])
def test_output_shape(N, C, H, W, params, oCH, oCW, oH, oW):
    x = nn.Variable((N, C, H, W))
    assert F.patch_correlation(x, x, **params).shape == (N, oCH, oCW, oH, oW)


@pytest.mark.parametrize("C", [1, 2, 5])
@pytest.mark.parametrize("H, W, params, output", [
    (3, 3, {'patch': 3, 'padding': 0}, np.array([[[[[9]]]]])),
    (2, 3, {'patch': (2, 3), 'padding': 0}, np.array([[[[[6]]]]])),
    (2, 3, {'patch': (2, 3), 'padding': (0, 1)}, np.array([[[[[4, 6, 4]]]]])),
    (3, 3, {'patch': 3, 'padding': (0, 0, 2, 0)}, np.array([[[[[3, 6, 9]]]]])),
    (3, 3, {'patch': 3, 'padding': (0, 0, 0, 2)}, np.array([[[[[9, 6, 3]]]]])),
])
def test_padding(C, H, W, params, output):
    x = F.constant(1, (1, C, H, W))
    y = F.patch_correlation(x, x, **params)
    y.forward()
    assert np.allclose(y.d, output * C)


def patch_correlation(x1, x2, patch=(1, 1), shift=(0, 0), patch_step=(1, 1),
                      shift_step=(1, 1), padding=(0, 0, 0, 0)):
    # inputs and output are NHWC tensors

    oh = (x1.shape[1] + sum(padding[:2]) -
          patch[0] + patch_step[0]) // patch_step[0]
    ow = (x1.shape[2] + sum(padding[2:]) -
          patch[1] + patch_step[1]) // patch_step[1]
    och = int(np.ceil((1 + 2 * shift[0]) / shift_step[0]))
    ocw = int(np.ceil((1 + 2 * shift[1]) / shift_step[1]))
    out_array = np.empty((x1.shape[0], oh, ow, och, ocw), dtype=x1.dtype)
    out_index = np.ndindex(out_array.shape)

    x1 = np.pad(x1, ((0, 0), padding[:2], padding[2:], (0, 0)), 'constant')
    x2 = np.pad(x2, ((0, 0), padding[:2], padding[2:], (0, 0)), 'constant')
    x2 = np.pad(x2, ((0, 0), *zip(shift, shift), (0, 0)), 'constant')

    for n in range(x1.shape[0]):
        for y in range(0, x1.shape[1] - patch[0] + 1, patch_step[0]):
            for x in range(0, x1.shape[2] - patch[1] + 1, patch_step[1]):
                p1 = x1[n, y:y+patch[0], x:x+patch[1]]
                for yy in range(y, y + 1 + 2 * shift[0], shift_step[0]):
                    for xx in range(x, x + 1 + 2 * shift[1], shift_step[1]):
                        p2 = x2[n, yy:yy+patch[0], xx:xx+patch[1]]
                        out_array[next(out_index)] = np.sum(p1 * p2)

    return out_array


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [314])
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("C", [1, 2])
@pytest.mark.parametrize("H, W, params", [
    (5, 5, {'patch': (1, 1), 'shift': (2, 2), 'shift_step': (2, 2)}),
    (4, 4, {'patch': (3, 3), 'padding': (1, 2, 2, 1)}),
    (5, 9, {'patch': (2, 3), 'shift': (1, 1), 'patch_step': (2, 3)}),
    (3, 8, {'patch': (1, 2), 'patch_step': (2, 3), 'padding': (1, 1, 0, 0)}),
    (5, 5, {'shift': (1, 0), 'shift_step': (1, 2), 'padding': (0, 0, 0, 0)}),
])
def test_forward_backward(N, C, H, W, params, seed, ctx, func_name):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(N, H, W, C).astype(np.float32)
    x2 = rng.randn(N, H, W, C).astype(np.float32)
    # must use function_bases.patch_correlation for func_name check in tester
    function_tester(rng, nn.function_bases.patch_correlation, patch_correlation,
                    [x1, x2], func_kwargs=params, func_name=func_name, ctx=ctx,
                    atol_b=2e-2)
