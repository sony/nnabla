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
    u = I.UniformInitializer((0, 2))
    inspecs.append([Inspec((64, 32, 224, 224), u)])
    return inspecs


@pytest.mark.parametrize('inspecs', inspecs_params())
@pytest.mark.parametrize('op',
                         ['logical_and_scalar', 'logical_or_scalar', 'logical_xor_scalar',
                          'greater_scalar', 'greater_equal_scalar',
                          'less_scalar', 'less_equal_scalar',
                          'equal_scalar', 'not_equal_scalar'])
def test_scalar_logical(inspecs, op, nnabla_opts):
    func = getattr(F, op)
    fb = FunctionBenchmark(
        func, inspecs, [1], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


@pytest.mark.parametrize('inspecs', inspecs_params())
def test_logical_not(inspecs, nnabla_opts):
    func = F.logical_not
    fb = FunctionBenchmark(
        func, inspecs, [], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


def pairwise_inspecs_params():
    inspecs = []
    u = I.UniformInitializer((0, 2))
    inspecs.append([Inspec((64, 32, 224, 224), u),
                    Inspec((64, 32, 224, 224), u)])
    return inspecs


@pytest.mark.parametrize('inspecs', pairwise_inspecs_params())
@pytest.mark.parametrize('op',
                         ['logical_and', 'logical_or', 'logical_xor',
                          'greater', 'greater_equal',
                          'less', 'less_equal',
                          'equal', 'not_equal'])
def test_pairwise_logical(inspecs, op, nnabla_opts):
    func = getattr(F, op)
    fb = FunctionBenchmark(
        func, inspecs, [], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
