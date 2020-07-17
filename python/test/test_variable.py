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
from nnabla.testing import assert_allclose


def test_manip():
    v = nn.Variable([2, 3, 4])
    assert v.shape == (2, 3, 4)
    with pytest.raises(Exception):
        v.reste_shape([1, 2])
    v.reset_shape([1, 2], force=True)
    assert v.shape == (1, 2)


@pytest.mark.parametrize("need_grad", [True, False])
def test_from_array(need_grad):
    data = np.random.randint(0, 10, size=(2, 3, 4))
    grad = np.random.randint(0, 10, size=(2, 3, 4))

    v1 = nn.Variable.from_numpy_array(data, need_grad=need_grad)
    assert np.all(v1.d == data)
    assert v1.d.dtype == data.dtype
    assert v1.need_grad == need_grad

    v2 = nn.Variable.from_numpy_array(data, grad, need_grad)
    assert np.all(v2.d == data)
    assert v2.d.dtype == data.dtype
    assert np.all(v2.g == grad)
    assert v2.g.dtype == grad.dtype
    assert v2.need_grad == need_grad


def test_data_grad_reference():
    v = nn.Variable([2, 3, 4])
    assert v.d.dtype == np.float32
    assert v.g.dtype == np.float32


def test_dtype_conversion():
    v = nn.Variable([2, 3, 4])
    a = v.data.cast(np.int)
    a[...] = 2
    assert (v.data.dtype == np.int)
    assert np.all(a == 2)
    b = v.data.cast(np.float32)
    assert b.dtype == np.float32
    assert b is not a
    assert np.all(b == 2)
    b[...] = np.random.randn(*b.shape) * 10
    c = v.data.cast(np.int32)
    assert np.all(c == b.astype(np.int32))


def test_data_grad():
    v = nn.Variable([2, 3, 4])
    v.d[...] = np.random.randn(*v.shape)
    assert v.d is not v.g
    assert not np.all(v.d == v.g)


def test_get_unlinked_variable():
    v = nn.Variable([2, 3, 4], need_grad=True)
    grad = np.random.randn(*v.shape).astype(np.float32)
    v.g = grad
    v.d = np.random.randn(*v.shape)
    import nnabla.functions as F
    with nn.context_scope(nn.Context()), nn.auto_forward():
        v2 = F.identity(v)
        v2_u = v2.get_unlinked_variable()
        assert v2_u.need_grad
        v3 = F.identity(v2_u)
    v2_u.grad.zero()
    v2_g = v2_u.g.copy()
    v3.backward(clear_buffer=False)
    assert type(v2_u) == type(v2)
    assert np.all(v.g == grad)
    assert np.all(v2_u.g == v2.g)
    assert np.all(v2_u.g == v2_g + 1)

    # Check need_grad option
    assert v2.get_unlinked_variable(need_grad=True).need_grad
    assert not v2.get_unlinked_variable(need_grad=False).need_grad


def test_reshape():
    v = nn.Variable([2, 3, 4], need_grad=True)
    grad = np.random.randn(*v.shape).astype(np.float32)
    v.g = grad
    v.d = np.random.randn(*v.shape)
    import nnabla.functions as F
    with nn.context_scope(nn.Context()), nn.auto_forward():
        v2 = F.identity(v)
        v2_s = v2.reshape((3, 4, 2))
        v3 = F.identity(v2_s)
    v3.backward(clear_buffer=False)
    assert np.all(v2_s.g.flat == v2.g.flat)
    assert np.all(v2_s.g == 1)
    v2.d = 1
    assert np.all(v2_s.d == 1)

    # Check unlink
    v2_su = v2.reshape((3, 4, 2), unlink=True)
    assert v2_su.need_grad
    assert v2_su.parent is None
    v2_su.need_grad = False
    v2_su2 = v2_su.reshape((3, 4, 2), unlink=True)
    assert not v2_su2.need_grad
    assert v2_su2.parent is None


def test_persistent():
    x = nn.Variable([2, 3, 4], need_grad=True)
    x1 = x + 1
    x2 = x1 + 1
    x3 = x2 + 1
    y = x3 + 1
    x3.persistent = True
    x.data.zero()
    y.forward(clear_buffer=True)
    assert_allclose(x3.d, 3)
    y.forward(clear_no_need_grad=True)
    y.backward(clear_buffer=True)
    assert_allclose(x3.d, 3)
    assert_allclose(x3.g, 1)


def test_name():
    x = nn.Variable([2, 3])
    x.name = "VariableName"
    assert x.name == "VariableName"


def test_name_all_variables():
    def net(h):
        import nnabla.functions as F
        import nnabla.parametric_functions as PF
        h = PF.convolution(h, 3, (3, 3), name="conv1")
        h = PF.batch_normalization(h, name="bn1")
        h = F.relu(h)
        h = F.max_pooling(h, (2, 2))
        h = PF.convolution(h, 3, (3, 3), name="conv2")
        h = PF.batch_normalization(h, name="bn2")
        pred = F.relu(h)
        return pred

    class Namer(object):
        def __init__(self, ):
            self.counter = 0

        def __call__(self, nnabla_func):
            for v in nnabla_func.outputs:
                v.name = "{}_output_{:05d}".format(
                    nnabla_func.name, self.counter)
                self.counter += 1

    class Confirmer(object):
        def __init__(self, ):
            self.counter = 0

        def __call__(self, nnabla_func):
            for v in nnabla_func.outputs:
                assert v.name == "{}_output_{:05d}".format(
                    nnabla_func.name, self.counter)
                self.counter += 1
    x = nn.Variable([2, 3, 8, 8])
    pred = net(x)
    pred.visit(Namer())
    pred.forward(clear_no_need_grad=True)
    pred.backward(clear_buffer=True)
    pred.visit(Confirmer())


def test_clear_all_graph_links():
    import nnabla.functions as F
    import nnabla.parametric_functions as PF

    class OneStepRNN(object):
        def __init__(self, batch_size=8, state_size=8):
            self.lstm0 = PF.LSTMCell(batch_size, state_size, name="lsmt0")
            self.lstm1 = PF.LSTMCell(batch_size, state_size, name="lsmt1")
            self.affine = PF.affine

        def __call__(self, x, n_class=10):
            h = self.lstm0(x)
            h = self.lstm1(h)
            h = self.affine(h, n_class)
            return h
    T = 3
    batch_size = 2
    dims = 4
    state_size = 8
    one_step_rnn = OneStepRNN(batch_size, state_size)
    # Forward: unroll over time
    loss = 0
    for t in range(T):
        x = nn.Variable.from_numpy_array(
            np.random.randn(batch_size, dims))
        y = nn.Variable.from_numpy_array(
            np.random.choice(np.arange(10), batch_size, replace=True)).reshape((batch_size, 1))
        pred = one_step_rnn(x)
        l = F.mean(F.softmax_cross_entropy(pred, y))
        loss += l
    loss /= T
    # Backward then truncate
    loss.backward()
    loss.clear_all_graph_links()

    assert one_step_rnn.lstm0.h.parent == None
    assert one_step_rnn.lstm0.c.parent == None
    assert one_step_rnn.lstm1.h.parent == None
    assert one_step_rnn.lstm1.c.parent == None


def test_function_references():
    import nnabla as nn
    import nnabla.parametric_functions as PF

    v = nn.Variable.from_numpy_array(np.random.randn(2, 4))

    assert len(v.function_references) == 0

    h1 = PF.affine(v, 10, name="affine1")

    assert len(v.function_references) == 1
    assert h1.parent in v.function_references

    h2 = PF.affine(v, 10, name="affine2")

    assert len(v.function_references) == 2
    assert h1.parent in v.function_references
    assert h2.parent in v.function_references

    del h1

    assert len(v.function_references) == 1
    assert h2.parent in v.function_references

    del h2

    assert len(v.function_references) == 0


@pytest.mark.parametrize("f", [lambda x: x, hash])
def test_variable_equality_and_hash(f):
    shape = (2, 3, 4)
    x = nn.Variable(shape)
    assert f(x) == f(x)

    y = nn.Variable(shape)
    assert f(x) != f(y)

    y = x.get_unlinked_variable()
    assert f(x) == f(y)

    y.need_grad = True
    assert f(x) == f(y)


def test_variable_set():
    # Testing hash and equality operator via set
    shape = (2, 3, 4)
    x = nn.Variable(shape)
    s = set()
    s.add(x)
    assert x in s

    y = nn.Variable(shape)
    assert y not in s

    y = x.get_unlinked_variable()
    assert y in s

    y.need_grad = True
    assert y in s


def test_prohibit_clear_data():
    import nnabla.functions as F
    nn.prefer_cached_array(False)
    shape = (2, 3, 4)
    var_np = np.random.rand(*shape)

    # the case of root variable
    x1 = nn.Variable.from_numpy_array(var_np)
    y1 = F.reshape(x1, (-1,), inplace=True)
    y1 = F.reshape(y1, shape, inplace=True) * 2

    x2 = nn.Variable.from_numpy_array(var_np)
    y2 = F.reshape(x2, (-1,), inplace=False)
    y2 = F.reshape(y2, shape, inplace=False) * 2

    nn.forward_all([y1, y2], clear_buffer=True)
    assert_allclose(x1.d, x2.d)
    assert_allclose(y1.d, y2.d)

    # the case of persistent variable
    x1 = nn.Variable.from_numpy_array(var_np)
    p_y1 = F.mul_scalar(x1, 2).apply(persistent=True)
    y1 = F.reshape(p_y1, (-1,), inplace=True)
    y1 = F.reshape(y1, shape, inplace=True) * 2

    x2 = nn.Variable.from_numpy_array(var_np)
    p_y2 = F.mul_scalar(x2, 2).apply(persistent=True)
    y2 = F.reshape(p_y2, (-1,), inplace=False)
    y2 = F.reshape(y2, shape, inplace=False) * 2

    nn.forward_all([y1, y2], clear_buffer=True)
    assert_allclose(p_y1.d, p_y2.d)
    assert_allclose(y1.d, y2.d)

    # the case of rewire_on root variable
    # graph A: x11 -> f_inplace -> y11
    x11 = nn.Variable.from_numpy_array(var_np)
    y11 = F.reshape(x11, (-1,), inplace=True)

    # graph B: x12 -> f_inplace -> mul_scalar -> y12
    x12 = nn.Variable(shape=y11.shape)
    y12 = F.reshape(x12, shape, inplace=True) * 2

    # graph A->B: x11 -> f_inplace -> f_inplace -> mul_scalar -> y12
    x12.rewire_on(y11)

    x2 = nn.Variable.from_numpy_array(var_np)
    y2 = F.reshape(x2, (-1,), inplace=False)
    y2 = F.reshape(y2, shape, inplace=False) * 2

    nn.forward_all([y12, y2], clear_buffer=True)
    assert_allclose(x11.d, x2.d)
    assert_allclose(y12.d, y2.d)

    # the case of rewire_on persistent variable
    # graph A: x11 -> mul_scalar -> p_x11 -> f_inplace -> y11
    x11 = nn.Variable.from_numpy_array(var_np)
    p_x11 = F.mul_scalar(x11, 2).apply(persistent=True)
    y11 = F.reshape(p_x11, (-1,), inplace=True)

    # graph B: x12 -> f_inplace -> mul_scalar -> y12
    x12 = nn.Variable(shape=y11.shape)
    y12 = F.reshape(x12, shape, inplace=True) * 2

    # graph A->B: ... -> p_x11 -> f_inplace -> f_inplace -> mul_scalar -> y12
    x12.rewire_on(y11)

    x2 = nn.Variable.from_numpy_array(var_np)
    p_x2 = F.mul_scalar(x2, 2).apply(persistent=True)
    y2 = F.reshape(p_x2, (-1,), inplace=False)
    y2 = F.reshape(y2, shape, inplace=False) * 2

    nn.forward_all([y12, y2], clear_buffer=True)
    assert_allclose(p_x11.d, p_x2.d)
    assert_allclose(y12.d, y2.d)
