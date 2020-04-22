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
import nnabla.solvers as S
import nnabla.utils.save
import nnabla.utils.load
from nnabla.ext_utils import get_extension_context


def test_save_load_parameters():
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
    nnabla.utils.load.load(['tmp.nnp'])


def check_save_load(tmpdir, x, y, variable_batch_size):

    contents = {
        'networks': [
            {'name': 'Validation',
             'batch_size': 1,
             'outputs': {'y': y},
             'names': {'x': x}}],
        'executors': [
            {'name': 'Runtime',
             'network': 'Validation',
             'data': ['x'],
             'output': ['y']}]}
    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join('tmp.nnp')
    nnp_file = tmppath.strpath
    nnabla.utils.save.save(nnp_file, contents,
                           variable_batch_size=variable_batch_size)
    nnabla.utils.load.load([nnp_file])


@pytest.mark.parametrize("variable_batch_size", [False, True])
@pytest.mark.parametrize("shape", [(10, 56, -1), (-1, 56, 7, 20, 10)])
def test_save_load_reshape(tmpdir, variable_batch_size, shape):
    nn.clear_parameters()
    x = nn.Variable([10, 1, 28, 28, 10, 10])
    y = F.reshape(x, shape=shape)
    check_save_load(tmpdir, x, y, variable_batch_size)


@pytest.mark.parametrize("variable_batch_size", [False, True])
def test_save_load_broadcast(tmpdir, variable_batch_size):
    nn.clear_parameters()
    x = nn.Variable([10, 1, 4, 1, 8])
    y = F.broadcast(x, shape=[10, 1, 4, 3, 8])
    check_save_load(tmpdir, x, y, variable_batch_size)


@pytest.mark.parametrize("datasets_o", [['dataset1', 'dataset2'], ('dataset1', 'dataset2'),
                                        'dataset1'])
@pytest.mark.parametrize("datasets_m", [['dataset1', 'dataset2'], ('dataset1', 'dataset2'),
                                        'dataset2'])
def test_save_load_multi_datasets(tmpdir, datasets_o, datasets_m):
    nn.clear_parameters()
    ctx = get_extension_context(
        'cpu', device_id=0, type_config='float')
    nn.set_default_context(ctx)

    batch_size = 64
    x = nn.Variable([batch_size, 1, 28, 28])
    Affine = PF.affine(x, 1, name='Affine')
    Sigmoid = F.sigmoid(Affine)

    target = nn.Variable([batch_size, 1])
    target.data.fill(1)
    BinaryCrossEntropy = F.binary_cross_entropy(Sigmoid, target)

    solver = S.Adam()
    solver.set_parameters(nn.get_parameters())
    solver.set_learning_rate(5e-4)

    contents = {
        'global_config': {
            'default_context': ctx
        },
        'training_config': {
            'max_epoch': 100,
            'iter_per_epoch': 23,
            'save_best': True,
            'monitor_interval': 10
        },
        'networks': [
            {
                'name': 'Main',
                'batch_size': batch_size,
                'outputs': {'BinaryCrossEntropy': BinaryCrossEntropy},
                'names': {'x': x}
            },
            {
                'name': 'MainValidation',
                'batch_size': batch_size,
                'outputs': {'BinaryCrossEntropy': BinaryCrossEntropy},
                'names': {'x': x}
            },
            {
                'name': 'MainRuntime',
                'batch_size': batch_size,
                'outputs': {'Sigmoid': Sigmoid},
                'names': {'x': x}
            }
        ],
        'datasets': [
            {
                'name': 'dataset1',
                'uri': 'DATASET_TRAINING1',
                'cache_dir': 'here_it_is',
                'shuffle': True,
                'batch_size': batch_size,
                'no_image_normalization': False,
                'variables': {'x': x, 'BinaryCrossEntropy': BinaryCrossEntropy}
            },
            {
                'name': 'dataset2',
                'uri': 'DATASET_TRAINING2',
                'cache_dir': 'here_it_is',
                'shuffle': True,
                'batch_size': batch_size,
                'no_image_normalization': False,
                'variables': {'x': x, 'BinaryCrossEntropy': BinaryCrossEntropy},
            }
        ],
        'optimizers': [
            {
                'name': 'optimizer',
                'solver': solver,
                'network': 'Main',
                'dataset': datasets_o,
                'weight_decay': 0,
                'lr_decay': 1,
                'lr_decay_interval': 1,
                'update_interval': 1
            }
        ],
        'monitors': [
            {
                'name': 'train_error',
                'network': 'MainValidation',
                'dataset': datasets_m
            },
            {
                'name': 'valid_error',
                'network': 'MainValidation',
                'dataset': datasets_m
            }
        ],
        'executors': [
            {
                'name': 'Executor',
                'network': 'MainRuntime',
                'data': ['x'],
                'output': ['Sigmoid']
            }
        ]
    }

    tmpdir.ensure(dir=True)
    tmppath = tmpdir.join('testsave.nnp')
    nnp_file = tmppath.strpath
    nnabla.utils.save.save(nnp_file, contents)
    nnabla.utils.load.load([nnp_file])


def test_save_load_with_file_object():
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
    import io
    nnpdata = io.BytesIO()
    nnabla.utils.save.save(nnpdata, contents, extension='.nnp')
    nnabla.utils.load.load(nnpdata, extension='.nnp')
