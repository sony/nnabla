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
import nnabla as nn
from nnabla.function import PythonFunction


class Add2(PythonFunction):

    def __init__(self, ctx=None):
        super(Add2, self).__init__(ctx)

    @property
    def name(self):
        return "PythonAdd2"

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        outputs[0].reset_shape(inputs[0].shape, True)

    def forward_impl(self, inputs, outputs):
        outputs[0].d = inputs[0].d + inputs[1].d

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        if propagate_down[0]:
            if accum[0]:
                inputs[0].g += outputs[0].g
            else:
                inputs[0].g = outputs[0].g
        if propagate_down[1]:
            if accum[1]:
                inputs[1].g += outputs[0].g
            else:
                inputs[1].g = outputs[0].g


@pytest.mark.parametrize("seed", [314])
def test_python_add2_forward_backward(seed):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.randn(2, 3, 4).astype(np.float32) * 2 for _ in range(2)]
    add2 = Add2()
    function_tester(rng, add2, lambda x, y: x + y, inputs, atol_b=2e-3)
