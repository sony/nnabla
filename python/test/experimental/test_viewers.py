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


def test_simple_graph(tmpdir):
    try:
        from graphviz import Digraph
    except:
        pytest.skip('Skip because graphviz is not installed.')
    from nnabla.experimental.viewers import SimpleGraph
    sg = SimpleGraph()

    nn.clear_parameters()
    x = nn.Variable((2, 3, 4, 4))
    with nn.parameter_scope('c1'):
        h = PF.convolution(x, 8, (3, 3), pad=(1, 1))
        h = F.relu(PF.batch_normalization(h))
    with nn.parameter_scope('f1'):
        y = PF.affine(h, 10)
    g = sg.create_graphviz_digraph(y)
    assert isinstance(g, Digraph)
    tmpdir.ensure(dir=True)
    # fpath = tmpdir.join('graph').strpath
    fpath = 'tmp-draw'
    sg.save(y, fpath, format='png')
