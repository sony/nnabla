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

import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF


def test_show_graph():
    try:
        from nnabla.experimental.tb_graph_writer import TBGraphWriter
    except:
        pytest.skip(
            'Skip because tensorboardX and tensorflow is not installed.')

    nn.clear_parameters()
    x = nn.Variable((2, 3, 4, 4))
    with nn.parameter_scope('c1'):
        h = PF.convolution(x, 8, (3, 3), pad=(1, 1))
        h = F.relu(PF.batch_normalization(h))
    with nn.parameter_scope('f1'):
        y = PF.affine(h, 10)

    with TBGraphWriter(log_dir='log_out') as tb:
        tb.from_variable(y, output_name="y")


def test_show_curve():
    try:
        from nnabla.experimental.tb_graph_writer import TBGraphWriter
    except:
        pytest.skip(
            'Skip because tensorboardX and tensorflow is not installed.')

    with TBGraphWriter(log_dir='log_out') as tb:
        values = []
        for i in range(360):
            s = np.sin(i / 180.0 * np.pi)
            tb.add_scalar("show_curve/sin", s, i)
            values.append(s)

        nd_values = np.array(values)
        for i in range(10):
            tb.add_histogram("histogram", nd_values, i)
            nd_values += 0.05
