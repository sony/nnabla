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


import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.utils.save


def test_generate_tmp_nnp():
    nn.clear_parameters()
    batch_size = 16
    x0 = nn.Variable([batch_size, 100])
    x1 = nn.Variable([batch_size, 100])
    h1_0 = PF.affine(x0, 100, name='affine1_0')
    h1_1 = PF.affine(x1, 100, name='affine1_0')
    h1 = F.tanh(h1_0 + h1_1)
    h2 = F.tanh(PF.affine(h1, 50, name='affine2'))
    y0 = PF.affine(h2, 10, name='affiney_0')
    y1 = PF.affine(h2, 10, name='affiney_1')

    contents = {
        'networks': [
            {'name': 'net1',
             'batch_size': batch_size,
             'outputs': {'y0': y0, 'y1': y1},
             'names': {'x0': x0, 'x1': x1}}],
        'executors': [
            {'name': 'runtime',
             'network': 'net1',
             'data': ['x0', 'x1'],
             'output': ['y0', 'y1']}]}
    nnabla.utils.save.save('tmp.nnp', contents)
