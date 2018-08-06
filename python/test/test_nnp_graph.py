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

from six.moves import range
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


@pytest.mark.parametrize("seed", [313])
def test_nnp_graph(seed):

    rng = np.random.RandomState(seed)

    def unit(i, prefix):
        c1 = PF.convolution(i, 4, (3, 3), pad=(1, 1), name=prefix + '-c1')
        c2 = PF.convolution(F.relu(c1), 4,
                            (3, 3), pad=(1, 1), name=prefix + '-c2')
        c = F.add2(c2, c1, inplace=True)
        return c
    x = nn.Variable([2, 3, 4, 4])
    c1 = unit(x, 'c1')
    c2 = unit(x, 'c2')
    y = PF.affine(c2, 5, name='fc')

    runtime_contents = {
        'networks': [
            {'name': 'graph',
             'batch_size': 2,
             'outputs': {'y': y},
             'names': {'x': x}}],
    }
    import tempfile
    tmpdir = tempfile.mkdtemp()
    import os
    nnp_file = os.path.join(tmpdir, 'tmp.nnp')
    try:
        from nnabla.utils.save import save
        save(nnp_file, runtime_contents)
        from nnabla.utils import nnp_graph
        nnp = nnp_graph.NnpLoader(nnp_file)
    finally:
        import shutil
        shutil.rmtree(tmpdir)
    graph = nnp.get_network('graph')
    x2 = graph.inputs['x']
    y2 = graph.outputs['y']

    d = rng.randn(*x.shape).astype(np.float32)
    x.d = d
    x2.d = d
    y.forward(clear_buffer=True)
    y2.forward(clear_buffer=True)
    from nbla_test_utils import ArrayDiffStats
    assert np.allclose(y.d, y2.d), str(ArrayDiffStats(y.d, y2.d))


def check_nnp_graph_save_load(tmpdir, x, y, batch_size, variable_batch_size):

    # Save
    contents = {
        'networks': [
            {'name': 'graph',
             'batch_size': 1,
             'outputs': {'y': y},
             'names': {'x': x}}]}
    from nnabla.utils.save import save
    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join('tmp.nnp')
    nnp_file = tmppath.strpath
    save(nnp_file, contents,
         variable_batch_size=variable_batch_size)

    # Load
    from nnabla.utils import nnp_graph
    nnp = nnp_graph.NnpLoader(nnp_file)
    graph = nnp.get_network('graph', batch_size=batch_size)
    x2 = graph.inputs['x']
    y2 = graph.outputs['y']
    if not variable_batch_size:
        assert x2.shape == x.shape
        assert y2.shape == y.shape
        return x2, y2

    assert x2.shape[0] == batch_size
    assert y2.shape[0] == batch_size
    return x2, y2


@pytest.mark.parametrize('variable_batch_size', [False, True])
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.parametrize("shape", [(10, 56, -1), (10, 56, 7, 20, 10)])
def test_nnp_graph_reshape(tmpdir, variable_batch_size, batch_size, shape):
    x = nn.Variable([10, 1, 28, 28, 10, 10])
    y = F.reshape(x, shape=shape)
    x2, y2 = check_nnp_graph_save_load(
        tmpdir, x, y, batch_size, variable_batch_size)
    if not variable_batch_size:
        return
    shape2 = list(y.shape)
    shape2[0] = batch_size
    x2.d = np.random.randn(*x2.shape)
    y2.forward()
    assert np.allclose(y2.d, x2.d.reshape(shape2))


@pytest.mark.parametrize('variable_batch_size', [False, True])
@pytest.mark.parametrize('batch_size', [1, 4])
def test_nnp_graph_broadcast(tmpdir, variable_batch_size, batch_size):
    x = nn.Variable([10, 1, 4, 1, 8])
    y = F.broadcast(x, shape=[10, 1, 4, 3, 8])
    x2, y2 = check_nnp_graph_save_load(
        tmpdir, x, y, batch_size, variable_batch_size)
