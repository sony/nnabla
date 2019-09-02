# Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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


import sys
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
import numpy as np

from nnabla.ext_utils import get_extension_context

from nnabla.utils.profiler import GraphProfiler, GraphProfilerCsvWriter


def cnn(x, n_class):
    c1 = PF.convolution(x, 16, (5, 5), name='conv1')
    c1 = F.relu(F.max_pooling(c1, (2, 2)), inplace=True)
    c2 = PF.convolution(c1, 16, (5, 5), name='conv2')
    c2 = F.relu(F.max_pooling(c2, (2, 2)), inplace=True)
    c3 = F.relu(PF.affine(c2, 50, name='fc3'), inplace=True)
    c4 = PF.affine(c3, n_class, name='fc4')

    return c4


def test_profiling():
    batch_size = 16
    n_class = 10

    device = "cpu"
    ctx = get_extension_context(device)
    nn.set_default_context(ctx)

    x = nn.Variable(shape=(batch_size, 1, 32, 32))
    t = nn.Variable(shape=(batch_size, 1))

    y = cnn(x, n_class)
    loss = F.mean(F.softmax_cross_entropy(y, t))

    solver = S.Sgd()
    solver.set_parameters(nn.get_parameters())

    x.d = np.random.normal(size=x.shape)
    t.d = np.floor(np.random.rand(*t.shape) *
                   (n_class - 0.000001)).astype(np.int32)

    B = GraphProfiler(loss, solver=solver, device_id=0,
                      ext_name=device, n_run=1000)

    B.run()

    csv_writer = GraphProfilerCsvWriter(gb=B, file=sys.stdout)
    csv_writer.write()
