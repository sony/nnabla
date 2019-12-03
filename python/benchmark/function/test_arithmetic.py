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


def inspecs_params():
    inspecs = []
    u = I.UniformInitializer((0.5, 1.0))
    inspecs.append([Inspec((64, 1000), u)])
    inspecs.append([Inspec((64, 32, 224, 224), u)])
    inspecs.append([Inspec((64, 128, 56, 56), u)])
    return inspecs


@pytest.mark.parametrize('inspecs', inspecs_params())
@pytest.mark.parametrize('op',
                         ['add_scalar', 'mul_scalar', 'pow_scalar',
                          'r_sub_scalar', 'r_div_scalar',
                          'r_pow_scalar', 'maximum_scalar',
                          'minimum_scalar'])
def test_scalar_arithmetic(inspecs, op, nnabla_opts):
    func = getattr(F, op)
    fb = FunctionBenchmark(
        func, inspecs, [0.5], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


def pairwise_inspecs_params():
    inspecs = []
    u = I.UniformInitializer((0.5, 1.0))
    inspecs.append([Inspec((64, 1000), u),
                    Inspec((64, 1000), u)])
    inspecs.append([Inspec((64, 32, 224, 224), u),
                    Inspec((64, 32, 224, 224), u)])
    inspecs.append([Inspec((64, 128, 56, 56), u),
                    Inspec((64, 128, 56, 56), u)])
    return inspecs


@pytest.mark.parametrize('inspecs', pairwise_inspecs_params())
@pytest.mark.parametrize('op',
                         ['add2', 'sub2', 'mul2', 'div2', 'pow2',
                          'maximum2', 'minimum2'])
def test_pairwise_arithmetic(inspecs, op, nnabla_opts):
    func = getattr(F, op)
    fb = FunctionBenchmark(
        func, inspecs, [], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
