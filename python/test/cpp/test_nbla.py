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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.utils.save
import nnabla.utils.load

import numpy as np
import os
from subprocess import check_call, call, check_output
import platform


def command_exists(command):
    which = 'which'
    if platform.system() == 'Windows':
        which = 'where'
    retval = call([which, command])
    return not retval


def check_nbla_infer(tmpdir, x, y, batch_size):
    if not command_exists('nbla'):
        pytest.skip('An executable `nbla` is not in path.')

    # A. save a created braph to nnp.
    contents = {
        'networks': [
            {'name': 'graph',
             'batch_size': 1,
             'outputs': {'y': y},
             'names': {'x': x}}],
        'executors': [
            {'name': 'runtime',
             'network': 'graph',
             'data': ['x'],
             'output': ['y']}
        ]}

    from nnabla.utils.save import save
    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join('tmp.nnp')
    nnp_file = tmppath.strpath
    save(nnp_file, contents)

    # B. Get result with nnp_graph
    from nnabla.utils import nnp_graph
    nnp = nnp_graph.NnpLoader(nnp_file)
    graph = nnp.get_network('graph', batch_size=batch_size)
    x2 = graph.inputs['x']
    y2 = graph.outputs['y']
    x2.d = np.random.randn(*x2.shape).astype(np.float32)
    y2.forward()

    # C. Get result with nbla
    input_bin = tmpdir.join('tmp_in.bin')
    input_bin_file = input_bin.strpath
    x2.d.tofile(input_bin_file)

    output_bin = tmpdir.join('tmp_out')
    check_call(['nbla', 'infer', '-e', 'runtime', '-b', str(batch_size),
                '-o', output_bin.strpath, nnp_file, input_bin_file])

    # D. Compare
    y3 = np.fromfile(output_bin.strpath + '_0.bin',
                     dtype=np.float32).reshape(y2.shape)
    assert np.allclose(y2.d, y3)


@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize("shape", [(10, 56, -1), (-1, 56, 7, 20, 10)])
def test_nbla_reshape(tmpdir, batch_size, shape):

    x = nn.Variable([10, 1, 28, 28, 10, 10])
    y = F.reshape(x, shape=shape)
    check_nbla_infer(tmpdir, x, y, batch_size)


@pytest.mark.parametrize('batch_size', [1, 4])
def test_nbla_broadcast(tmpdir, batch_size):

    x = nn.Variable([10, 1, 4, 1, 5])
    y = F.broadcast(x, shape=[10, 8, 4, 1, 5])
    check_nbla_infer(tmpdir, x, y, batch_size)
