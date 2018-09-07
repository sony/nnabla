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

ctxs = list_context('Interpolate')


def ref_interpolate(x, scale, output_size, mode, align_corners):
    import math
    assert mode == 'linear'
    if scale is None and output_size is None:
        raise ValueError('either must be given')
    elif output_size is None:
        assert len(scale) == 2, 'Only 2D supported.'
        output_size = [int(math.floor(s * d))
                       for s, d in zip(scale, x.shape[-len(scale):])]
    else:
        assert len(output_size) == 2, 'Only 2D supported.'

    # 2D bilinear
    oh, ow = output_size
    ih, iw = x.shape[-2:]
    outer_shape = x.shape[:-2]
    x = x.reshape(-1, ih, iw)

    def get_scale(src, dst, align_corners):
        if dst == 1:
            return 0
        if align_corners:
            return float(src - 1) / (dst - 1)
        return float(src) / dst

    sy = get_scale(ih, oh, align_corners)
    sx = get_scale(iw, ow, align_corners)

    # Output index
    oy = np.arange(output_size[0])
    ox = np.arange(output_size[1])

    # Input index in float
    if align_corners:
        fy = oy * sy
        fx = ox * sx
    else:
        fy = np.maximum(0, sy * (oy + 0.5) - 0.5)
        fx = np.maximum(0, sx * (ox + 0.5) - 0.5)

    # Input index
    iy = fy.astype(np.int32)
    ix = fx.astype(np.int32)
    iyp1 = np.minimum(iy + 1, ih - 1)
    ixp1 = np.minimum(ix + 1, iw - 1)
    ly1 = (fy - iy)[None, :, None]
    ly0 = 1.0 - ly1
    lx1 = (fx - ix)[None, None, :]
    lx0 = 1.0 - lx1
    iz = np.arange(x.shape[0])

    val0 = lx0 * x[np.ix_(iz, iy, ix)]
    val1 = lx1 * x[np.ix_(iz, iy, ixp1)]
    val2 = lx0 * x[np.ix_(iz, iyp1, ix)]
    val3 = lx1 * x[np.ix_(iz, iyp1, ixp1)]
    ret = ly0 * (val0 + val1)
    ret += ly1 * (val2 + val3)
    return ret.reshape(outer_shape + (oh, ow))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape", [(2, 3, 4, 4), (3, 3)])
@pytest.mark.parametrize("outsize, scale", [
    ((8, 6), None),
    ((2, 1), None),
    (None, (2.5, 1)),
    (None, (0.5, 0.5)), ])
@pytest.mark.parametrize('align_corners', [False, True])
@pytest.mark.parametrize('mode', ['linear'])
@pytest.mark.parametrize("seed", [313])
def test_interpolate_forward_backward(seed, inshape, outsize, scale, align_corners, mode, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    # Generate ND inputs
    inputs = [rng.randn(*inshape).astype(np.float32)]
    function_tester(rng, F.interpolate, ref_interpolate, inputs, func_args=[
                    scale, outsize, mode, align_corners], atol_f=1e-6, atol_b=1e-2, ctx=ctx, func_name=func_name)
