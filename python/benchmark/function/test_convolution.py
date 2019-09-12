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


def conv_params():
    inputs = [Inspec((64, 3, 224, 224))]
    func_kwargs = dict(
        outmaps=64,
        kernel=(7, 7),
        pad=(3, 3),
        stride=(2, 2),
        with_bias=False)
    list_params = [(inputs, func_kwargs)]
    inputs = [Inspec((64, 64, 56, 56))]
    func_kwargs = dict(
        outmaps=64,
        kernel=(3, 3),
        pad=(1, 1),
        stride=(1, 1),
        with_bias=False)
    list_params.append((inputs, func_kwargs))
    inputs = [Inspec((64, 512, 7, 7))]
    func_kwargs = dict(
        outmaps=512,
        kernel=(3, 3),
        pad=(1, 1),
        stride=(1, 1),
        with_bias=False)
    list_params.append((inputs, func_kwargs))
    return list_params


@pytest.mark.parametrize('inputs, func_kwargs', conv_params())
def test_convolution(inputs, func_kwargs, nnabla_opts):
    fb = FunctionBenchmark(
        PF.convolution, inputs, [], func_kwargs,
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
