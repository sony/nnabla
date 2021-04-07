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

# You can use this script like...
#
# [float]
# > pytest --nnabla-ext=cudnn --nnabla-ext-device-id=0 test_binary_operation.py
#
# [half]
# > pytest --nnabla-ext=cudnn --nnabla-ext-device-id=0 \
#   --nnabla-ext-type-config=half test_binary_operation.py


# The following input shapes of Add2 are uniquely extracted from ResNet50
# of batch size 5 with various normalization schemes (batch normalization,
# group normalization, layer normalization, and instance normalization),
# with channel-last or not, and with weight standardization for each
# convolution weight or not.
def resnet50_inspecs_params_without_broadcast():
    inspecs = []
    u = I.UniformInitializer((0.5, 1.0))

    inspecs.append([Inspec((5, 2048, 7, 7), u),
                    Inspec((5, 2048, 7, 7), u)])
    inspecs.append([Inspec((5, 1024, 14, 14), u),
                    Inspec((5, 1024, 14, 14), u)])
    inspecs.append([Inspec((5, 512, 28, 28), u),
                    Inspec((5, 512, 28, 28), u)])
    inspecs.append([Inspec((5, 256, 56, 56), u),
                    Inspec((5, 256, 56, 56), u)])
    inspecs.append([Inspec((5, 56, 56, 256), u),
                    Inspec((5, 56, 56, 256), u)])
    inspecs.append([Inspec((5, 28, 28, 512), u),
                    Inspec((5, 28, 28, 512), u)])
    inspecs.append([Inspec((5, 14, 14, 1024), u),
                    Inspec((5, 14, 14, 1024), u)])
    inspecs.append([Inspec((5, 7, 7, 2048), u),
                    Inspec((5, 7, 7, 2048), u)])

    return inspecs


def resnet50_inspecs_params_with_broadcast():
    inspecs = []
    u = I.UniformInitializer((0.5, 1.0))

    inspecs.append([Inspec((5, 1024, 14, 14), u),
                    Inspec((1, 1024, 1, 1), u)])
    inspecs.append([Inspec((5, 1024, 14, 14), u),
                    Inspec((1, 1024, 14, 14), u)])
    inspecs.append([Inspec((5, 112, 112, 64), u),
                    Inspec((1, 1, 1, 64), u)])
    inspecs.append([Inspec((5, 112, 112, 64), u),
                    Inspec((1, 112, 112, 64), u)])
    inspecs.append([Inspec((5, 128, 28, 28), u),
                    Inspec((1, 128, 1, 1), u)])
    inspecs.append([Inspec((5, 128, 28, 28), u),
                    Inspec((1, 128, 28, 28), u)])
    inspecs.append([Inspec((5, 128, 56, 56), u),
                    Inspec((1, 128, 1, 1), u)])
    inspecs.append([Inspec((5, 128, 56, 56), u),
                    Inspec((1, 128, 56, 56), u)])
    inspecs.append([Inspec((5, 14, 14, 1024), u),
                    Inspec((1, 1, 1, 1024), u)])
    inspecs.append([Inspec((5, 14, 14, 1024), u),
                    Inspec((1, 14, 14, 1024), u)])
    inspecs.append([Inspec((5, 14, 14, 256), u),
                    Inspec((1, 1, 1, 256), u)])
    inspecs.append([Inspec((5, 14, 14, 256), u),
                    Inspec((1, 14, 14, 256), u)])
    inspecs.append([Inspec((5, 14, 14, 512), u),
                    Inspec((1, 1, 1, 512), u)])
    inspecs.append([Inspec((5, 14, 14, 512), u),
                    Inspec((1, 14, 14, 512), u)])
    inspecs.append([Inspec((5, 2048, 7, 7), u),
                    Inspec((1, 2048, 1, 1), u)])
    inspecs.append([Inspec((5, 2048, 7, 7), u),
                    Inspec((1, 2048, 7, 7), u)])
    inspecs.append([Inspec((5, 256, 14, 14), u),
                    Inspec((1, 256, 1, 1), u)])
    inspecs.append([Inspec((5, 256, 14, 14), u),
                    Inspec((1, 256, 14, 14), u)])
    inspecs.append([Inspec((5, 256, 28, 28), u),
                    Inspec((1, 256, 1, 1), u)])
    inspecs.append([Inspec((5, 256, 28, 28), u),
                    Inspec((1, 256, 28, 28), u)])
    inspecs.append([Inspec((5, 256, 56, 56), u),
                    Inspec((1, 256, 1, 1), u)])
    inspecs.append([Inspec((5, 256, 56, 56), u),
                    Inspec((1, 256, 56, 56), u)])
    inspecs.append([Inspec((5, 28, 28, 128), u),
                    Inspec((1, 1, 1, 128), u)])
    inspecs.append([Inspec((5, 28, 28, 128), u),
                    Inspec((1, 28, 28, 128), u)])
    inspecs.append([Inspec((5, 28, 28, 256), u),
                    Inspec((1, 1, 1, 256), u)])
    inspecs.append([Inspec((5, 28, 28, 256), u),
                    Inspec((1, 28, 28, 256), u)])
    inspecs.append([Inspec((5, 28, 28, 512), u),
                    Inspec((1, 1, 1, 512), u)])
    inspecs.append([Inspec((5, 28, 28, 512), u),
                    Inspec((1, 28, 28, 512), u)])
    inspecs.append([Inspec((5, 512, 14, 14), u),
                    Inspec((1, 512, 1, 1), u)])
    inspecs.append([Inspec((5, 512, 14, 14), u),
                    Inspec((1, 512, 14, 14), u)])
    inspecs.append([Inspec((5, 512, 28, 28), u),
                    Inspec((1, 512, 1, 1), u)])
    inspecs.append([Inspec((5, 512, 28, 28), u),
                    Inspec((1, 512, 28, 28), u)])
    inspecs.append([Inspec((5, 512, 7, 7), u),
                    Inspec((1, 512, 1, 1), u)])
    inspecs.append([Inspec((5, 512, 7, 7), u),
                    Inspec((1, 512, 7, 7), u)])
    inspecs.append([Inspec((5, 56, 56, 128), u),
                    Inspec((1, 1, 1, 128), u)])
    inspecs.append([Inspec((5, 56, 56, 128), u),
                    Inspec((1, 56, 56, 128), u)])
    inspecs.append([Inspec((5, 56, 56, 256), u),
                    Inspec((1, 1, 1, 256), u)])
    inspecs.append([Inspec((5, 56, 56, 256), u),
                    Inspec((1, 56, 56, 256), u)])
    inspecs.append([Inspec((5, 56, 56, 64), u),
                    Inspec((1, 1, 1, 64), u)])
    inspecs.append([Inspec((5, 56, 56, 64), u),
                    Inspec((1, 56, 56, 64), u)])
    inspecs.append([Inspec((5, 64, 112, 112), u),
                    Inspec((1, 64, 1, 1), u)])
    inspecs.append([Inspec((5, 64, 112, 112), u),
                    Inspec((1, 64, 112, 112), u)])
    inspecs.append([Inspec((5, 64, 56, 56), u),
                    Inspec((1, 64, 1, 1), u)])
    inspecs.append([Inspec((5, 64, 56, 56), u),
                    Inspec((1, 64, 56, 56), u)])
    inspecs.append([Inspec((5, 7, 7, 2048), u),
                    Inspec((1, 1, 1, 2048), u)])
    inspecs.append([Inspec((5, 7, 7, 2048), u),
                    Inspec((1, 7, 7, 2048), u)])
    inspecs.append([Inspec((5, 7, 7, 512), u),
                    Inspec((1, 1, 1, 512), u)])
    inspecs.append([Inspec((5, 7, 7, 512), u),
                    Inspec((1, 7, 7, 512), u)])

    return inspecs

################################################################################
# Add2
################################################################################


def add2_inspecs_params():
    inspecs = resnet50_inspecs_params_without_broadcast()
    inspecs += resnet50_inspecs_params_with_broadcast()
    return inspecs


@pytest.mark.parametrize('inspecs', add2_inspecs_params())
@pytest.mark.parametrize('op', ['add2'])
def test_add2_with_broadcast(inspecs, op, nnabla_opts):
    func = getattr(F, op)
    fb = FunctionBenchmark(
        func, inspecs, [], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


################################################################################
# Mul2
################################################################################
def mul2_inspecs_params():
    inspecs += resnet50_inspecs_params_with_broadcast()
    return inspecs


@pytest.mark.parametrize('inspecs', mul2_inspecs_params())
@pytest.mark.parametrize('op', ['mul2'])
def test_mul2_with_broadcast(inspecs, op, nnabla_opts):
    func = getattr(F, op)
    fb = FunctionBenchmark(
        func, inspecs, [], {},
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
