# Copyright 2021 Sony Corporation.
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


@pytest.mark.parametrize('B', [1])
@pytest.mark.parametrize('R', [512, 1024, 2048])
@pytest.mark.parametrize('N', [512, 1024, 2048])
@pytest.mark.parametrize('axis', [1, 2])
@pytest.mark.parametrize('exclusive', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_cumprod(B, R, N, axis, exclusive, reverse, nnabla_opts):
    # Create input
    x_shape = (B, R, N, 1)

    def init(shape):
        rng = np.random.RandomState(123)
        x = rng.randn(*shape)
        # Make zero elements with the probability of one zero-element per scan axis.
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
