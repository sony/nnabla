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

ctxs = list_context('Pad')


def ref_pad(x, pad_with, mode, constant_value):
    pair_len = int(len(pad_with)/2)
    pad_list = [(0, 0)] * (len(x.shape)-pair_len)
    pad_list.extend([(a, b) for a, b in zip(pad_with[:-1:2], pad_with[1::2])])
    new_pad = tuple(pad_list)

    ret = np.pad(x, new_pad, mode, constant_values=constant_value)
    return ret


def pad_forward_backward(seed, inshape, pad_with, mode, constant_value, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Generate ND inputs
    i = rng.randn(*inshape).astype(np.float32)
    inputs_nd = [i]

    function_tester(rng, F.pad, ref_pad, inputs_nd, func_args=[
                    pad_with, mode, constant_value], ctx=ctx, func_name=func_name, dstep=1e-1)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape_1d", [(4,), (5,)])
@pytest.mark.parametrize("pad_with_1d", [(2, 2), (1, 1), (2, 3)])
@pytest.mark.parametrize("mode, constant_value", [('constant', 0.0), ('constant', 0.2), ('constant', 5.5), ('constant', -0.1)])
@pytest.mark.parametrize("seed", [313])
def test_pad_forward_backward_1D(seed, inshape_1d, pad_with_1d, mode, constant_value, ctx, func_name):
    pad_forward_backward(seed, inshape_1d, pad_with_1d,
                         mode, constant_value, ctx, func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape_2d", [(4, 4), (5, 5), (2, 3)])
@pytest.mark.parametrize("pad_with_2d", [(2, 2), (1, 1), (2, 3), (2, 2, 2, 2), (3, 3, 3, 3), (2, 3, 3, 4)])
@pytest.mark.parametrize("mode, constant_value", [('constant', 0.0), ('constant', 0.2), ('constant', 5.5), ('constant', -0.1)])
@pytest.mark.parametrize("seed", [313])
def test_pad_forward_backward_2D(seed, inshape_2d, pad_with_2d, mode, constant_value, ctx, func_name):
    pad_forward_backward(seed, inshape_2d, pad_with_2d,
                         mode, constant_value, ctx, func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape_Nd", [(2, 2, 2), (3, 5, 7), (2, 3, 2), (2, 2, 2, 2), (3, 5, 7, 1), (2, 3, 4, 6), (2, 2, 2, 2, 2), (3, 1, 5, 7, 3), (2, 3, 1, 1, 4, 5), (2, 2, 2, 2, 2, 2), (3, 3, 1, 5, 7, 3), (2, 2, 3, 1, 1, 4, 5)])
@pytest.mark.parametrize("pad_with_3d", [(2, 2), (1, 1), (2, 3), (2, 2, 2, 2), (3, 3, 3, 3), (2, 3, 3, 4), (2, 2, 2, 2, 2, 2), (3, 3, 3, 3, 3, 3), (2, 3, 2, 3, 3, 4)])
@pytest.mark.parametrize("mode, constant_value", [('constant', 0.0), ('constant', 0.2), ('constant', 5.5), ('constant', -0.1)])
@pytest.mark.parametrize("seed", [313])
def test_pad_forward_backward_Nd(seed, inshape_Nd, pad_with_3d, mode, constant_value, ctx, func_name):
    pad_forward_backward(seed, inshape_Nd, pad_with_3d,
                         mode, constant_value, ctx, func_name)
