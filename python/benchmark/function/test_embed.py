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
import nnabla.parametric_functions as PF

from function_benchmark import FunctionBenchmark, Inspec


def embed_params():
    """(inspecs, n_inputs, n_features)"""
    params = []
    params.append(([Inspec((64, ), I.UniformIntInitializer(
        (0, 40000)), need_grad=False)], 40000, 256))
    return params


@pytest.mark.parametrize('inspecs, n_inputs, n_features', embed_params())
def test_embed(inspecs, n_inputs, n_features, nnabla_opts):
    fb = FunctionBenchmark(
        PF.embed, inspecs, [],
        dict(n_inputs=n_inputs, n_features=n_features),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
