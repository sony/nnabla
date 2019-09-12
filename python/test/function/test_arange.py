# Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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
from nbla_test_utils import list_context, function_tester

ctxs = list_context('Arange')


def ref_arange(start, stop, step):
    return np.arange(start, stop, step).astype(np.float32)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("start, stop, step", [
    (0, 10, 1),
    (0, 10, -1),
    (10, 0, -1),
    (0, 10, 11),
    (0, 10, 0.5),
    (0, 10, 0.3),
    (0, -10, -1),
    (-9.9, 9.9, 1.1),
    (9.9, -9.9, -1.1),
])
def test_arange_forward(start, stop, step, ctx, func_name):
    function_tester(None, F.arange, ref_arange, inputs=[], ctx=ctx,
                    func_args=[start, stop, step],
                    func_name=func_name, backward=[])
