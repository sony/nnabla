# Copyright 2022 Sony Group Corporation.
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


def inspecs_params():
    inspecs = []
    inspecs.append(([Inspec((64, 80, 224, 224))], 1))
    inspecs.append(([Inspec((64, 224, 224, 80))], 3))
    inspecs.append(([Inspec((64, 20, 224, 224))], 1))
    inspecs.append(([Inspec((64, 224, 224, 20))], 3))
    inspecs.append(([Inspec((1, 1000))], 1))
    inspecs.append(([Inspec((64, 1000))], 1))
    inspecs.append(([Inspec((768, 50, 50))], 2))
    return inspecs


@pytest.mark.parametrize('inspecs, axis', inspecs_params())
def test_softmax(inspecs, axis, nnabla_opts):
    fb = FunctionBenchmark(
        F.softmax, inspecs, [], dict(axis=axis),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
