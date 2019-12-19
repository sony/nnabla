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

import nnabla.initializer as I
import nnabla.functions as F

from function_benchmark import FunctionBenchmark, Inspec


@pytest.mark.parametrize("seed", [313])
def pad_params():
    inspecs = []
    u = I.UniformInitializer((0.5, 1.0))
    inspecs.append([Inspec((2, 2, 2, 2), u)])
    inspecs.append([Inspec((2, 3, 2, 3), u)])
    inspecs.append([Inspec((2, 20, 200, 200), u)])
    return inspecs


@pytest.mark.parametrize('inspecs', pad_params())
def test_pad(inspecs, nnabla_opts):
    fb = FunctionBenchmark(
        F.pad, inspecs, [(10, 10, 10, 10), 'constant', 0.0], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
