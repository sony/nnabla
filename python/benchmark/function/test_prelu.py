# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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
import nnabla.parametric_functions as PF

from function_benchmark import FunctionBenchmark, Inspec


def inspecs_params():
    inspecs = []
    inspecs.append([Inspec((64, 64, 224, 224))])
    inspecs.append([Inspec((64, 128, 112, 112))])
    inspecs.append([Inspec((64, 512, 14, 14))])
    return inspecs


@pytest.mark.parametrize('inspecs', inspecs_params())
@pytest.mark.parametrize('shared', [True, False])
def test_activation(inspecs, shared, nnabla_opts):
    fb = FunctionBenchmark(
        PF.prelu, inspecs, [1], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
