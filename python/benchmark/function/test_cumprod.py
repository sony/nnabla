# Copyright 2021 Sony Group Corporation.
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

from function_benchmark import FunctionBenchmark, Inspec


class Case:
    def __init__(self, shape, axis, rtol=1e-6):
        # rtol (relative tolerance) 1e-6 is default for assert_allclose
        self.shape = shape
        self.axis = axis
        self.rtol = rtol

    # Print this message by pytest when a test fails.
    def __repr__(self):
        return 'Case(shape=' + str(self.shape) + \
               ' axes=' + str(self.axis) + \
               ', rtol=' + str(self.rtol) + ')'


test_cases = [
    # --------------------------------
    # Common use case
    # --------------------------------
    # Axis 0
    Case((512, 512), 0),
    Case((512, 1024), 0),
    Case((512, 2048), 0),
    Case((1024, 512), 0),
    Case((1024, 1024), 0),
    Case((1024, 2048), 0),
    Case((2048, 512), 0),
    Case((2048, 1024), 0),
    Case((2048, 2048), 0),
    # Axis 1
    Case((512, 512), 1),
    Case((512, 1024), 1),
    Case((512, 2048), 1),
    Case((1024, 512), 1),
    Case((1024, 1024), 1),
    Case((1024, 2048), 1),
    Case((2048, 512), 1),
    Case((2048, 1024), 1),
    Case((2048, 2048), 1),

    # --------------------------------
    # Large cases
    # --------------------------------
    Case((1024*1024, 32), 1),
    Case((32, 1024*1024), 0),
    Case((2048, 2048), 1),
    Case((2048, 2048), 0),
    Case((2024*2024, 2), 0),
    Case((2, 2024*2024), 1),

    # Weak cases
    # PyTorch uses Cub library in these cases.
    Case((2024*2024, 1), 0),
    Case((1, 2024*2024), 1),
]


@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("test_case", test_cases)
@pytest.mark.parametrize('exclusive', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize("with_mask", [True, False])
def test_cumprod(seed, test_case, exclusive, reverse, with_mask, nnabla_opts):
    x_shape = test_case.shape
    axis = test_case.axis

    def init(shape):
        rng = np.random.RandomState(seed)
        x = rng.randn(*shape)
        if with_mask:
            # Make zero elements with the probability of `1 / x_shape[axis]`.
            # It is the probability of existence of one zero element in each scan axis.
            mask = rng.rand(*shape) > (1.0 / shape[axis])
            x = x * mask
        return x
    need_grad = True

    inputs = [Inspec(x_shape, init, need_grad)]

    func_kwargs = dict(
        axis=axis,
        exclusive=exclusive,
        reverse=reverse,
    )
    fb = FunctionBenchmark(
        F.cumprod, inputs, [], func_kwargs,
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
