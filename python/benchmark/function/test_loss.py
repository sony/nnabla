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


def classification_inspecs_params():
    inspecs = []
    ui = I.UniformIntInitializer
    inspecs.append([Inspec((64, 1000)),
                    Inspec((64, 1), ui((0, 1000)), False)])
    inspecs.append([Inspec((64, 32, 224, 224)),
                    Inspec((64, 1, 224, 224), ui((0, 32)), False)])
    inspecs.append([Inspec((64, 128, 56, 56)),
                    Inspec((64, 1, 56, 56), ui((0, 128)), False)])
    return inspecs


@pytest.mark.parametrize('inspecs', classification_inspecs_params())
@pytest.mark.parametrize('loss',
                         ['softmax_cross_entropy',
                          'categorical_cross_entropy',
                          'top_n_error'])
def test_categorical_classification_loss(inspecs, loss, nnabla_opts):
    func = getattr(F, loss)
    fb = FunctionBenchmark(
        func, inspecs, [], dict(axis=1),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


# ============================================================================
def pairwise_inspecs_params(label_init=I.UniformIntInitializer((0, 2))):
    inspecs = []
    inspecs.append([Inspec((64, 1000)),
                    Inspec((64, 1000), label_init, False)])
    inspecs.append([Inspec((64, 32, 224, 224)),
                    Inspec((64, 32, 224, 224), label_init, False)])
    inspecs.append([Inspec((64, 128, 56, 56)),
                    Inspec((64, 128, 56, 56), label_init, False)])
    return inspecs


@pytest.mark.parametrize('inspecs', pairwise_inspecs_params())
@pytest.mark.parametrize('loss',
                         ['sigmoid_cross_entropy',
                          'binary_cross_entropy'])
def test_binary_classification_loss(inspecs, loss, nnabla_opts):
    func = getattr(F, loss)
    fb = FunctionBenchmark(
        func, inspecs, [], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


@pytest.mark.parametrize('inspecs',
                         pairwise_inspecs_params(I.UniformInitializer((0, 1))))
@pytest.mark.parametrize('loss',
                         ['squared_error',
                          'huber_loss',
                          'kl_multinomial'])
def test_pairwise_loss(inspecs, loss, nnabla_opts):
    func = getattr(F, loss)
    fb = FunctionBenchmark(
        func, inspecs, [], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
# ============================================================================
