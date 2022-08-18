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
    # Reported bad performance cases
    # These three cases were optimized well by cuTENSOR.
    inspecs.append(([Inspec((32, 144, 28, 1))], (0, 1, 3, 2)))
    inspecs.append(([Inspec((32, 144, 28, 3))], (0, 1, 3, 2)))
    inspecs.append(([Inspec((768, 50, 50))], (0, 2, 1)))

    # From ResNet-50
    # Input side
    inspecs.append(([Inspec((192, 224, 224, 3))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((192, 3, 224, 224))], (0, 2, 3, 1)))
    # Output side
    inspecs.append(([Inspec((192, 3, 3, 512))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((192, 512, 3, 3))], (0, 2, 3, 1)))
    inspecs.append(([Inspec((192, 1, 1, 2048))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((192, 2048, 1, 1))], (0, 2, 3, 1)))
    # Single input
    inspecs.append(([Inspec((1, 224, 224, 3))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((1, 3, 224, 224))], (0, 2, 3, 1)))
    inspecs.append(([Inspec((1, 3, 3, 512))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((1, 512, 3, 3))], (0, 2, 3, 1)))
    inspecs.append(([Inspec((1, 1, 1, 2048))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((1, 2048, 1, 1))], (0, 2, 3, 1)))

    # Other
    # 2D
    inspecs.append(([Inspec((64, 64))], (1, 0)))
    inspecs.append(([Inspec((1024, 1024))], (1, 0)))
    # 4D
    inspecs.append(([Inspec((64, 64, 64, 64))], (0, 1, 2, 3)))
    inspecs.append(([Inspec((64, 64, 64, 64))], (0, 1, 3, 2)))
    inspecs.append(([Inspec((64, 64, 64, 64))], (0, 3, 2, 1)))
    inspecs.append(([Inspec((64, 64, 64, 64))], (0, 2, 1, 3)))
    inspecs.append(([Inspec((64, 64, 64, 64))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((64, 64, 64, 64))], (0, 2, 3, 1)))
    # 4D misaligned
    inspecs.append(([Inspec((65, 65, 65, 65))], (0, 1, 2, 3)))
    inspecs.append(([Inspec((65, 65, 65, 65))], (0, 1, 3, 2)))
    inspecs.append(([Inspec((65, 65, 65, 65))], (0, 3, 2, 1)))
    inspecs.append(([Inspec((65, 65, 65, 65))], (0, 2, 1, 3)))
    inspecs.append(([Inspec((65, 65, 65, 65))], (0, 3, 1, 2)))
    inspecs.append(([Inspec((65, 65, 65, 65))], (0, 2, 3, 1)))

    return inspecs


@pytest.mark.parametrize('inspecs, axis', inspecs_params())
def test_transpose(inspecs, axis, nnabla_opts):
    fb = FunctionBenchmark(
        F.transpose, inspecs, [axis], dict(),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
