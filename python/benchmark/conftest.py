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
import os


def pytest_addoption(parser):
    parser.addoption('--nnabla-ext', type=str, default='cpu',
                     help='Extension path, e.g. "cpu", "cuda", "cudnn".')
    parser.addoption('--nnabla-ext-type-config', type=str, default='float',
                     help='Extension type-config, e.g. "float", "half".')
    parser.addoption('--nnabla-ext-device-id', type=str, default='0',
                     help='Keyward argument `device_id` of extensions specifies'
                     ' gpu index where test runs')
    parser.addoption('--nnabla-benchmark-output', type=str,
                     default='benchmark-output',
                     help='Pass a directory path to where benchmark log will outputs.')


@pytest.fixture(scope='session')
def nnabla_opts(request):
    """Parse options and expose as a fixture.

    Returns: NNablaOpts
        An  object which has ext, ext_kwargs, benchmark_output_dir and
        function_benchmark_writer as attributes.
    """
    from collections import namedtuple
    from function_benchmark import FunctionBenchmarkCsvWriter
    getoption = request.config.getoption
    ext = getoption("--nnabla-ext")
    ext_kwargs = dict(device_id=getoption("--nnabla-ext-device-id"),
                      type_config=getoption('--nnabla-ext-type-config'))
    benchmark_output_dir = getoption('--nnabla-benchmark-output')
    if not os.path.isdir(benchmark_output_dir):
        os.makedirs(benchmark_output_dir)
    function_benchmark_name = 'function_{}_{}_{}.csv'.format(
        ext.replace('.', '-'), ext_kwargs['device_id'], ext_kwargs['type_config'])
    function_benchmark_file = os.path.join(
        benchmark_output_dir, function_benchmark_name)
    function_benchmark_writer = FunctionBenchmarkCsvWriter(
        open(function_benchmark_file, 'w'))

    NNablaOpts = namedtuple("NNablaOpts",
                            ['ext',
                             'ext_kwargs',
                             'benchmark_output_dir',
                             'function_benchmark_writer'])
    return NNablaOpts(ext, ext_kwargs,
                      benchmark_output_dir, function_benchmark_writer)
