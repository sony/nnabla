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

from six import iterkeys

import numpy as np
import nnabla.parametric_functions as PF
from nnabla.initializer import UniformInitializer


def test_get_set_pop_parameter():
    import nnabla as nn
    from nnabla.parameter import set_parameter, pop_parameter, get_parameter
    nn.clear_parameters()
    x = nn.Variable((2, 3, 4, 5))
    key = 'a/b/c'
    set_parameter(key, x)
    x2 = get_parameter(key)
    assert x is x2
    x3 = pop_parameter(key)
    assert x is x3
    x4 = get_parameter(key)
    assert x4 is None


def test_get_parameter_or_create_need_grad():
    """Testing if need_grad flag works not not.
    """
    import nnabla as nn
    from nnabla.parameter import get_parameter_or_create
    nn.clear_parameters()
    key1 = 'p/param1'
    param1 = get_parameter_or_create(key1, (2, 3, 4, 5), need_grad=True)
    p1d = np.random.randn(*param1.shape).astype(np.float32)
    p1g = np.random.randn(*param1.shape).astype(np.float32)
    param1.d = p1d
    param1.g = p1g
    param1_f = get_parameter_or_create(
        key1, param1.shape, need_grad=False)
    assert not nn.get_parameters(grad_only=False)[key1].need_grad
    param1_f = get_parameter_or_create(
        key1, param1.shape, need_grad=True)
    assert nn.get_parameters()[key1].need_grad
    assert np.all(param1.d == p1d)
    assert np.all(param1.d == param1_f.d)
    param1.d = 1
    assert np.all(param1_f.d == 1)
    param1_f2 = get_parameter_or_create(
        key1, param1.shape, need_grad=True, as_need_grad=False)
    assert not param1_f2.need_grad
    nn.clear_parameters()


def test_get_parameter_with_initializer():
    """Testing with initializer
    """
    import nnabla as nn
    from nnabla.parameter import get_parameter_or_create
    nn.clear_parameters()
    rng = np.random.RandomState(seed=313)
    shape = (8, 8, 3, 3)

    # Instance inherited from BaseInitializer
    initializer = UniformInitializer(lim=(-1, 1), rng=rng)
    param1 = get_parameter_or_create(
        'param1', shape, initializer=initializer, need_grad=True)
    assert np.min(param1.d > -1) and np.max(param1.d < 1)

    # Numpy array
    initializer = rng.randn(*shape)
    param2 = get_parameter_or_create(
        'param2', initializer=initializer, need_grad=True)
    np.allclose(initializer, param2.d)

    # Random
    param3 = get_parameter_or_create('param3', shape, need_grad=True)

    nn.clear_parameters()


def test_parameter_scope_slash():
    """Testing if parameter_scope('aaa/bbb') works.
    """
    import nnabla as nn
    from nnabla.parameter import get_parameter_or_create
    nn.clear_parameters()
    with nn.parameter_scope('aaa/bbb'):
        param = get_parameter_or_create('ccc', (2, 3, 4, 5))
    ref = np.random.randn(*param.shape).astype(np.float32)
    param.d = ref

    with nn.parameter_scope('aaa'):
        with nn.parameter_scope('bbb'):
            param = get_parameter_or_create('ccc', (2, 3, 4, 5))
    assert np.all(param.d == ref)
    nn.clear_parameters()


# Dummy parametric function for test_parametric_function
@PF.parametric_function_api("dummy")
def dummy_parametric_function(shape, f=10, i=1, s="dummy",
                              fix_parameters=False):
    """Doc"""
    from nnabla import Variable
    from nnabla.parameter import get_parameter_or_create
    from nnabla.initializer import UniformInitializer
    p1 = get_parameter_or_create("p1", shape, UniformInitializer((-1, 1)))
    p2 = get_parameter_or_create(
        "p2", shape + (1,), UniformInitializer((-1, 1)))
    return Variable(shape)


def test_parametric_function_api():
    """
    Testing :function:`nnabla.parametric_functions.parametric_function_api`.
    """
    import nnabla as nn
    import inspect
    nn.clear_parameters()
    shape = (2, 3, 4)

    # Signature check
    spec = inspect.getargspec(dummy_parametric_function)
    assert spec.args == ['shape', 'f', 'i', 's', 'fix_parameters', 'name']
    assert spec.defaults == (10, 1, 'dummy', False, None)
    assert dummy_parametric_function.__doc__.splitlines()[0] == 'Doc'

    # Verify two different ways does the same thing.
    # Using name argument
    v = dummy_parametric_function(shape, name='group1')
    # Using parameter_scope
    with nn.parameter_scope('group1'):
        v = dummy_parametric_function(shape)

    params = nn.get_parameters()
    assert len(params) == 2
    assert list(iterkeys(params)) == ['group1/dummy/p1', 'group1/dummy/p2']

    # No scope
    v = dummy_parametric_function(shape)

    params = nn.get_parameters()
    len(params) == 4
    assert list(iterkeys(params)) == ['group1/dummy/p1', 'group1/dummy/p2',
                                      'dummy/p1', 'dummy/p2']
    nn.clear_parameters()


def test_parameter_as_need_grad():

    import nnabla as nn
    import nnabla.parametric_functions as PF
    import nnabla as nn

    nn.clear_parameters()
    x = nn.Variable((2, 5))
    y = PF.batch_normalization(x, fix_parameters=True)
    params = nn.get_parameters(grad_only=False)
    assert list(params.keys()) == [
        'bn/' + name for name in ['beta', 'gamma', 'mean', 'var']]
    assert params['bn/beta'].need_grad
    assert params['bn/gamma'].need_grad
    assert not params['bn/mean'].need_grad
    assert not params['bn/var'].need_grad

    assert not any([v.need_grad for v in y.parent.inputs[1:5]])
