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

from refs import (generate_transformation_2d, generate_transformation_3d,
                  affine_grid_2d, affine_grid_3d)

ctxs = list_context('AffineGrid')


def create_inputs(rng, batch_size, size):
    if len(size) == 2:
        theta = generate_transformation_2d(rng, batch_size)
    elif len(size) == 3:
        theta = generate_transformation_3d(rng, batch_size)
    return theta


def ref_affine_grid(theta, size, align_corners):
    if len(size) == 2:
        grid_s = affine_grid_2d(theta, size, align_corners)
    elif len(size) == 3:
        grid_s = affine_grid_3d(theta, size, align_corners)
    return grid_s


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("size", [(3, 3), (4, 4), (3, 4), (4, 3),
                                  (2, 3, 4), (4, 2, 3), (3, 4, 2), (3, 3, 3), (4, 4, 4)])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_affine_grid_forward_backward(seed, ctx, func_name, align_corners, size, batch_size):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    theta = create_inputs(rng, batch_size, size)
    inputs = [theta]
    func_args = [size, align_corners]
    function_tester(rng, F.affine_grid, ref_affine_grid, inputs, func_args=func_args,
                    ctx=ctx, func_name=func_name, backward=[True], atol_b=1e-2, atol_accum=1e-2)
