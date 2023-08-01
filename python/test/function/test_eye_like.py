# Copyright 2023 Sony Group Corporation.
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
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('EyeLike')


def ref_eye_like(x, k):
    y = np.eye(x.shape[0], x.shape[1], k=k, dtype=x.dtype)
    return y


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("inshape", [
    (3, 3),
    (3, 10),
    (10, 3),
])
@pytest.mark.parametrize("k", [
    10,
    2,
    0,
    -2,
    -10,
])
@pytest.mark.parametrize("seed", [313])
def test_eye_like_forward(seed, inshape, k, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(*inshape).astype(np.float32)]
    backward = [False]
    func_args = [k]
    function_tester(rng, F.eye_like, ref_eye_like, inputs,
                    func_name=func_name, func_args=func_args,
                    atol_f=0.0, ctx=ctx, backward=backward)
