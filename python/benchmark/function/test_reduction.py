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


import nnabla.functions as F

from function_benchmark import FunctionBenchmark, Inspec


def reduction_inspecs_params():
    inspecs = []
    inspecs.append([Inspec((2048, 8192 * 16))])
    inspecs.append([Inspec((2048, 8192 * 2))])
    inspecs.append([Inspec((64, 1000))])
    inspecs.append([Inspec((64, 32, 224, 224))])
    inspecs.append([Inspec((64, 512, 7, 7))])
    return inspecs


@pytest.mark.parametrize('inspecs', reduction_inspecs_params())
@pytest.mark.parametrize('reduction', ['sum', 'mean', 'max', 'min', 'prod'])
@pytest.mark.parametrize('axis', [None, 1])
def test_reduction_axis(inspecs, reduction, axis, nnabla_opts):
    func = getattr(F, reduction)
    fb = FunctionBenchmark(
        func, inspecs, [], dict(axis=axis),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
