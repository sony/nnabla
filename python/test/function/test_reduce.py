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
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('Reduce')


def ref_reduce(x, op):
    return eval('np.%s' % op)(x)


@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("op", ['mean', 'sum'])
@pytest.mark.parametrize("ctx, func_name", ctxs)
def test_reduce_forward_backward(seed, ctx, func_name, op):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2]
    function_tester(rng, F.reduce, ref_reduce, inputs,
                    func_args=[op], ctx=ctx)
