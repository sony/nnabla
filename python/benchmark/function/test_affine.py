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
import nnabla.parametric_functions as PF

from function_benchmark import FunctionBenchmark, Inspec


def affine_params():
    """(inspecs, n_outmaps)"""
    params = []
    params.append(([Inspec((128, 1, 28, 28))], 64))
    params.append(([Inspec((64, 512, 7, 7))], 4096))
    params.append(([Inspec((64, 4096))], 1000))
    params.append(([Inspec((64, 512))], 1000))
    return params


@pytest.mark.parametrize('inspecs, n_outmaps', affine_params())
def test_affine(inspecs, n_outmaps, nnabla_opts):
    fb = FunctionBenchmark(
        PF.affine, inspecs, [], dict(n_outmaps=n_outmaps),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
