# Copyright 2021 Sony Group Corporation.
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


import nnabla as nn
import nnabla.parametric_functions as PF
from nnabla.ext_utils import get_extension_context

from function_benchmark import FunctionBenchmark, Inspec


# The following benchmark cases are used for the works of optimizing normalization functions.


def general_cases(channel_last):
    '''
    Generate use cases
    '''
    inspec_and_axis = []

    batch = 16
    base_ch = 192
    ch_mul = [1, 1, 2, 2, 4, 4]
    channels = [base_ch * factor for factor in ch_mul]
    resolutions = [256, 128, 64, 32, 16, 8]
    axis = 3 if channel_last else 1

    # Create `inspec_and_axis`
    for ch, res in zip(channels, resolutions):
        if channel_last:
            shape = (batch, res, res, ch)
        else:
            shape = (batch, ch, res, res)
        inspec_and_axis.append(([Inspec(shape)], axis))

    return inspec_and_axis


def large_reduction_cases_for_group_norm(channel_last):
    '''
    Cases for checking large reduction size for group normalization
    '''
    inspec_and_axis = []

    batch = 1
    base_ch = 256
    ch_mul = [1, 2, 4, 8, 16, 32]
    channels = [base_ch * factor for factor in ch_mul]
    res = 128
    axis = 3 if channel_last else 1

    # Create `inspec_and_axis`
    for ch in channels:
        if channel_last:
            shape = (batch, res, res, ch)
        else:
            shape = (batch, ch, res, res)
        inspec_and_axis.append(([Inspec(shape)], axis))
    return inspec_and_axis


def large_reduction_cases_for_layer_norm():
    '''
    Cases for checking large reduction size for layer normalization
    '''
    # Same as group norm but only `channel_last == False` since `F.layer_norm()` does not have `axis` argument.
    return large_reduction_cases_for_group_norm(False)


def large_reduction_cases_for_instance_norm(channel_last):
    '''
    Cases for checking large reduction size for instance normalization
    '''
    inspec_and_axis = []

    batch = 1
    ch = 256
    res_mul = [1, 2, 4, 8]
    base_res = 128
    resolutions = [base_res * factor for factor in res_mul]
    axis = 3 if channel_last else 1

    # Create `inspec_and_axis`
    for res in resolutions:
        if channel_last:
            shape = (batch, res, res, ch)
        else:
            shape = (batch, ch, res, res)
        inspec_and_axis.append(([Inspec(shape)], axis))
    return inspec_and_axis


@pytest.mark.parametrize('inspec_and_axis', general_cases(False) +
                         general_cases(True) +
                         large_reduction_cases_for_group_norm(False) +
                         large_reduction_cases_for_group_norm(True))
@pytest.mark.parametrize('num_groups', [32])
def test_group_normalization(inspec_and_axis, num_groups, nnabla_opts):
    inspec, axis = inspec_and_axis
    fb = FunctionBenchmark(
        PF.group_normalization, inspec, [num_groups], dict(channel_axis=axis),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


@pytest.mark.parametrize('inspec_and_axis', general_cases(False) +
                         large_reduction_cases_for_layer_norm())
def test_layer_normalization(inspec_and_axis, nnabla_opts):
    inspec, axis = inspec_and_axis
    fb = FunctionBenchmark(
        PF.layer_normalization, inspec, [], dict(),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)


@pytest.mark.parametrize('inspec_and_axis', general_cases(False) +
                         general_cases(True) +
                         large_reduction_cases_for_instance_norm(False) +
                         large_reduction_cases_for_instance_norm(True))
def test_instance_normalization(inspec_and_axis, nnabla_opts):
    inspec, axis = inspec_and_axis
    fb = FunctionBenchmark(
        PF.instance_normalization, inspec, [], dict(channel_axis=axis),
        nnabla_opts.ext, nnabla_opts.ext_kwargs)
    fb.benchmark()
    fb.write(writer=nnabla_opts.function_benchmark_writer)
