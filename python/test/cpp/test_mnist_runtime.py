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
import nnabla.utils.save
import nnabla.utils.load
from nnabla.testing import assert_allclose

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


@pytest.mark.parametrize('batch_size', [1])
def test_examples_cpp_mnist_runtime(tmpdir, nnabla_examples_root, batch_size):
    pytest.skip('Temporarily skip due to mnist training data server trouble.')
    nn.clear_parameters()

    # A. Check this test can run
    if not nnabla_examples_root.available:
        pytest.skip('`nnabla-examples` can not be found.')

    if not command_exists('mnist_runtime'):
        pytest.skip('An executable `mnist_runtime` is not in path.')

    tmpdir.chdir()

    # B. Run mnist training.
    script = os.path.join(nnabla_examples_root.path,
                          'mnist-collection', 'classification.py')
    check_call(['python', script, '-i', '100'])

    # C. Get mnist_runtime results.
    nnp_file = tmpdir.join('tmp.monitor', 'lenet_result.nnp').strpath
    assert os.path.isfile(nnp_file)
    pgm_file = os.path.join(os.path.dirname(__file__),
                            '../../../examples/cpp/mnist_runtime/1.pgm')
    assert os.path.isfile(pgm_file)
    output = check_output(['mnist_runtime', nnp_file, pgm_file, 'Runtime'])
    output.decode('ascii').splitlines()[1].split(':')[1].strip()
    cpp_result = np.asarray(output.decode('ascii').splitlines()[1].split(':')[
                            1].strip().split(' '), dtype=np.float32)

    # D. Get nnp_graph results and compare.
    from nnabla.utils import nnp_graph
    nnp = nnp_graph.NnpLoader(nnp_file)
    graph = nnp.get_network('Validation', batch_size=batch_size)
    x = graph.inputs['x']
    y = graph.outputs['y']
    from nnabla.utils.image_utils import imread
    img = imread(pgm_file, grayscale=True)
    x.d = img
    y.forward()
    assert_allclose(y.d.flatten(), cpp_result)
