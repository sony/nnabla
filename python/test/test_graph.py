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
from nnabla.testing import assert_allclose, clear_called_flag_recorder


@pytest.mark.parametrize("seed", [313])
def test_graph_logreg(seed):
    rng = np.random.RandomState(seed)
    x = nn.Variable([2, 3, 4], need_grad=True)
    w = nn.Variable([12, 5], need_grad=True)
    b = nn.Variable([5], need_grad=True)
    t = nn.Variable([2, 1])
    x.d = rng.randn(*x.shape)
    w.d = rng.randn(*w.shape)
    b.d = rng.randn(*b.shape)
    t.d = rng.randint(0, 5, size=t.shape)

    nn.set_default_context(nn.Context())

    # Forwardprop by definition
    with nn.auto_forward():
        z = F.affine(x, w, b, 1)
        l = F.softmax_cross_entropy(z, t, 1)
        L = F.mean(l)

    # Backprop
    # Diff should be initialized since they are always accumulated
    x.g = 0
    w.g = 0
    b.g = 0
    L.backward(clear_buffer=True)
    x.g = rng.randn(*x.shape)

    inputs = [x, w, b]

    from nbla_test_utils import \
        compute_analytical_and_numerical_grad_graph as grads
    agrad, ngrad = grads(L, inputs, 1e-3)
    assert_allclose(ngrad, agrad, atol=1e-2)


@pytest.mark.parametrize("seed", [311])
@pytest.mark.parametrize("model", ["mlp", "recurrent", "convolution"])
def test_graph_model(model, seed):
    np.random.seed(313)
    rng = np.random.RandomState(seed)
    x = nn.Variable([2, 3, 4, 4], need_grad=True)
    t = nn.Variable([2, 1])
    x.d = rng.randn(*x.shape)
    t.d = rng.randint(0, 5, size=t.shape)

    nn.set_default_context(nn.Context())

    # Forwardprop by definition
    nn.clear_parameters()
    if model == "mlp":
        with nn.parameter_scope('fc1'):
            z = PF.affine(x, 3)
        z2 = F.relu(z, inplace=True)
        with nn.parameter_scope('fc2'):
            z3 = PF.affine(z2, 5)
    elif model == "recurrent":
        with nn.parameter_scope('fc1'):
            z = PF.affine(x, 8)
            z2 = F.relu(z, inplace=True)
        h = z2
        for _ in range(2):
            with nn.parameter_scope('fc2'):
                h = PF.affine(h, 8)
                h = F.relu(h, inplace=True)
        with nn.parameter_scope('fc3'):
            z3 = PF.affine(h, 5)
    elif model == "convolution":
        with nn.parameter_scope('conv1'):
            z = PF.convolution(x, 3, (2, 2))
            z2 = F.relu(z, inplace=True)
        with nn.parameter_scope('fc2'):
            z3 = PF.affine(z2, 5)
    else:
        raise ValueError()
    l = F.softmax_cross_entropy(z3, t, 1)
    L = F.mean(l)

    # Forwardprop
    L.forward(clear_no_need_grad=True)

    # Backprop
    # Diff should be initialized since they are always accumulated
    x.grad.zero()
    L.backward(clear_buffer=True)
    x.g = rng.randn(*x.shape)
    parameters = nn.get_parameters()
    for param in parameters.values():
        param.grad.zero()
    inputs = [x] + list(parameters.values())

    from nbla_test_utils import \
        compute_analytical_and_numerical_grad_graph as grads
    agrad, ngrad = grads(L, inputs, 1e-3)
    assert_allclose(ngrad, agrad, atol=1.05e-2)


@pytest.mark.parametrize("seed", [311])
def test_graph_unlink_backward(seed):
    rng = np.random.RandomState(seed)
    x0 = nn.Variable([2, 4], need_grad=True)
    x1 = nn.Variable([2, 4], need_grad=True)
    x0.d = rng.randn(*x0.shape)
    x1.d = rng.randn(*x1.shape)
    x0.grad.zero()
    x1.grad.zero()
    with nn.auto_forward():
        with nn.parameter_scope("fc0"):
            h0 = PF.affine(x0, 2)
        with nn.parameter_scope("fc1"):
            h1 = PF.affine(x1, 2)
        h0.need_grad = False
        h = h0 + h1
        with nn.parameter_scope("fc"):
            y = PF.affine(h, 1)
    y.backward(clear_buffer=True)
    assert np.all(x0.g == 0)
    assert not np.all(x1.g == 0)


@pytest.mark.parametrize("seed", [311])
def test_graph_clear_buffer(seed):
    np.random.seed(313)
    rng = np.random.RandomState(seed)
    x = nn.Variable([2, 3, 4, 4])
    t = nn.Variable([2, 1])
    x.d = rng.randn(*x.shape)
    t.d = rng.randint(0, 5, size=t.shape)

    # Network definition
    nn.set_default_context(nn.Context())
    nn.clear_parameters()
    x1 = x + 1
    x2 = x1 - 1
    with nn.parameter_scope('conv1'):
        z = PF.convolution(x2, 3, (2, 2))
        z2 = F.relu(z, inplace=True)
    with nn.parameter_scope('fc2'):
        z3 = PF.affine(z2, 5)
    l = F.softmax_cross_entropy(z3, t, 1)
    L = F.mean(l)

    # Forwardprop
    import tempfile
    import os
    tmpd = tempfile.mkdtemp()
    nn.save_parameters(os.path.join(tmpd, 'parameter.h5'))
    first = False
    for cnng in [False, True]:
        for cb in [False, True]:
            _ = nn.load_parameters(os.path.join(tmpd, 'parameter.h5'))
            for v in nn.get_parameters().values():
                v.grad.zero()
            L.forward(clear_no_need_grad=cnng)
            L.backward(clear_buffer=cb)
            if not first:
                first = True
                g = list(nn.get_parameters().values())[0].g.copy()
            else:
                g2 = list(nn.get_parameters().values())[0].g.copy()
                import platform
                if platform.machine() == 'ppc64le':
                    pytest.skip("This test fails on ppc64le")
                assert np.all(g == g2)


@pytest.mark.parametrize("seed", [311])
@pytest.mark.parametrize("clear_buffer", [True, False])
def test_graph_rewire(seed, clear_buffer):
    nn.clear_parameters()

    # A. defining graph definition utility
    def mlp2(x, scope):
        with nn.parameter_scope(scope):
            h = F.tanh(PF.affine(x, 10, name='a1'))
            h = F.tanh(PF.affine(h, 10, name='a1'))
            return h

    # A. Create a graph A.
    xa = nn.Variable((2, 10), need_grad=True)
    ya = mlp2(xa, 'a')

    # B. Create a graph B.
    xb = nn.Variable((2, 10), need_grad=True)
    yb = mlp2(xb, 'b')

    # C. Create directly connected graph.
    xc = nn.Variable((2, 10))
    yc = mlp2(mlp2(xc, 'a'), 'b')

    # D. Rewire the graphs A and B.
    xb.rewire_on(ya)

    # E. Check whether the results are the same.
    rng = np.random.RandomState(seed)
    data = rng.randn(*xa.shape)
    xa.d = data
    xc.d = data
    params = nn.get_parameters()

    def zero_grad():
        for p in params.values():
            p.grad.zero()

    def backup_params():
        return [p.g.copy() for p in params.values()]
    # Checking forward
    yb.forward(clear_no_need_grad=clear_buffer)
    yc.forward(clear_no_need_grad=clear_buffer)
    assert_allclose(yb.d, yc.d)
    # Checking backward
    zero_grad()
    yb.backward(clear_buffer=clear_buffer)
    gb = backup_params()
    zero_grad()
    yc.backward(clear_buffer=clear_buffer)
    gc = backup_params()
    assert_allclose(xa.d, xc.d)
    for b, c in zip(gb, gc):
        assert_allclose(b, c)


def test_deleted_outputs():
    rng = np.random.RandomState(313)

    x = nn.Variable((2, 3, 4, 5))
    h, m, v = PF.batch_normalization(x, output_stat=True)
    del m
    x.d = rng.randn(*x.shape).astype(np.float32)
    h.forward()
    h.backward()


def test_function_hook():
    '''
    Testing function hooks in forward and backward
    '''

    x = nn.Variable.from_numpy_array(
        np.zeros((2, 3), dtype=np.float32)).apply(need_grad=True)
    x.grad.zero()
    h = x + 2
    h.data.zero()
    h.grad.zero()
    y = h * 0.5
    y.data.zero()

    def forward_pre_hook(f):
        assert_allclose(f.outputs[0].d, 0)

    def forward_post_hook(f):
        if f.info.type_name == 'AddScalar':
            assert_allclose(f.outputs[0].d, 2)
        if f.info.type_name == 'MulScalar':
            assert_allclose(f.outputs[0].d, 1)

    def backward_pre_hook(f):
        assert_allclose(f.inputs[0].g, 0)

    def backward_post_hook(f):
        # Both h and x grad will be 0.5
        assert_allclose(f.inputs[0].g, 0.5)

    y.forward(function_pre_hook=forward_pre_hook,
              function_post_hook=forward_post_hook)
    y.backward(function_pre_hook=backward_pre_hook,
               function_post_hook=backward_post_hook)

    x.grad.zero()
    z = x * 0.1
    # Just calling test
    nn.forward_all((y, z), function_pre_hook=lambda f: None,
                   function_post_hook=lambda f: None)


@pytest.mark.parametrize("seed", [313])
def test_shared_variable_on_same_function(seed):
    rng = np.random.RandomState(313)
    xd = rng.randn(2, 3)
    x = nn.Variable.from_numpy_array(xd).apply(need_grad=True)
    x.grad.zero()
    y = x * x * x
    y.forward()
    y.backward()
    assert_allclose(x.g, 3 * xd ** 2)


@pytest.mark.parametrize("seed", [313])
def test_function_context(seed):
    rng = np.random.RandomState(313)
    xd = rng.randn(2, 3)
    x = nn.Variable.from_numpy_array(xd)
    ctx1 = nn.Context(backend=['cpu:float'],
                      array_class='CpuCachedArray', device_id='1')

    with nn.context_scope(ctx1):
        y = F.relu(x)
    ctx0 = nn.Context(backend=['cpu:float'],
                      array_class='CpuCachedArray', device_id='0')

    # TODO: use id or hash if we determine the spec
    assert str(ctx0) != str(ctx1)
    assert str(ctx1) == str(y.parent.context)

    with nn.context_scope(y.parent.context):
        z = F.relu(x)
    assert str(y.parent.context) == str(z.parent.context)


def test_no_need_grad_backward():
    '''
    This tests a previously existing bug where an
    intermediate variable with need_grad=False yet required
    to compute a gradient in a function has been unexpectedly cleared.
    '''
    nn.prefer_cached_array(False)
    x = nn.Variable(tuple(), need_grad=False)
    y = nn.Variable(tuple(), need_grad=True)
    z = nn.Variable(tuple(), need_grad=False)
    xx = x * 1
    yy = y * 1
    zz = z * 1
    a = xx * 3
    b = xx * yy
    c = xx * zz
    d = a * b * c

    x.data.fill(1)
    y.data.fill(2)
    z.data.fill(0.5)

    hook = None  # lambda f: print(f, list(map(lambda x: x.d, f.inputs)))
    d.forward(clear_no_need_grad=True, function_pre_hook=hook)
    y.grad.zero()
    d.backward(clear_buffer=True, function_pre_hook=hook)

    assert np.isclose(y.g, 1.5)


@pytest.mark.parametrize("clear_buffer", [False, True])
def test_no_need_grad_forward(clear_buffer):
    '''
    This tests a previously existing bug where an intermediate variable
    has been unexpectedly cleared before the end of life if
    it is used in an in-place function and
    another function at the same time.
    '''
    import nnabla as nn
    import nnabla.functions as F
    nn.prefer_cached_array(False)

    x = nn.Variable(tuple(), need_grad=False)
    xx = x * 1
    a = xx.reshape(x.shape)
    b = xx * 1
    d = a * b

    x.data.fill(1)

    d.forward(clear_no_need_grad=True, clear_buffer=clear_buffer)
    assert np.isclose(d.d, 1.0)


def test_no_need_grad_forward_double():
    '''
    This tests a previously existing bug where a variable used
    twice by a single function caused an unexpected clear due to
    incorrect count of function references.
    '''
    import nnabla as nn
    import nnabla.functions as F
    nn.prefer_cached_array(False)

    x = nn.Variable(tuple())
    xx = x * 1
    y = xx * xx
    z = xx * 1
    a = y * z
    x.data.fill(1)
    a.forward(clear_no_need_grad=True)
    assert np.isclose(a.d, 1.0)


class TestClearInput():

    def check_input_data_clear_called_flags(self, answer):
        result = clear_called_flag_recorder.get_input_clear_called_flags()
        assert len(result) == len(answer)
        for i, flags in enumerate(answer):
            assert len(result[i]) == len(flags)
            for j, flag in enumerate(flags):
                assert flag == result[i][j][0]

    def setup_method(self):
        clear_called_flag_recorder.activate_clear_called_flag_recorder()

    def teardown_method(self):
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

    # Test for clearing input in a network of two layers.
    def test_clear_input_if_no_need_grad0(self):
        x1 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)

        answer = []
        answer.append([False])
        answer.append([True])

        y1.forward(clear_no_need_grad=True)

        self.check_input_data_clear_called_flags(answer)

    # Test for clearing input in a network of three layers.
    def test_clear_input_if_no_need_grad1(self):
        x1 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)
        y2 = F.add_scalar(y1)

        answer = []
        answer.append([False])
        answer.append([True])
        answer.append([True])

        y2.forward(clear_no_need_grad=True)

        self.check_input_data_clear_called_flags(answer)

    # Test the case where an input is not cleared when it is required for backward at the previous layer function.
    def test_clear_input_if_no_need_grad2(self):
        x1 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)  # (1)
        y1 = F.tanh(xx1)  # (2)
        y2 = F.add_scalar(y1)  # (3)

        answer = []
        answer.append([False])
        answer.append([True])
        answer.append([False])
        # y1 must not be clear after (3) because y1 is required for backward of (2).

        y2.forward(clear_no_need_grad=True)

        self.check_input_data_clear_called_flags(answer)

    # Test for inplaced variable in a network of two layers.
    def test_clear_input_if_no_need_grad_inplace0(self):
        x1 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1, inplace=True)

        answer = []
        answer.append([False])
        answer.append([False])

        y1.forward(clear_no_need_grad=True)

        self.check_input_data_clear_called_flags(answer)

    # Test for inplaced variable in a network of three layers.
    def test_clear_input_if_no_need_grad_inplace1(self):
        x1 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1, inplace=True)
        y2 = F.add_scalar(y1)

        answer = []
        answer.append([False])
        answer.append([False])
        answer.append([False])

        y2.forward(clear_no_need_grad=True)
        self.check_input_data_clear_called_flags(answer)

    # Test for a variable shared with two layer functions.
    # Check if it is cleared after the both functions finish to use it.
    def test_clear_input_if_no_need_grad_branch0(self):
        x1 = nn.Variable([1, 5], need_grad=True)
        x2 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)  # (1)
        y2 = F.add_scalar(xx1)  # (2)
        y3 = F.add2(y1, y2)  # (3)

        answer = []
        answer.append([False])
        answer.append([False])  # (1) does not clear xx1
        answer.append([True])  # (2) clears xx1
        answer.append([True, True])

        y3.forward(clear_no_need_grad=True)
        self.check_input_data_clear_called_flags(answer)

    # Test for a variable shared with mul2 and add2.
    # add2 does not require it as input for backward, but mul2 does.
    def test_clear_input_if_no_need_grad_branch1(self):
        x1 = nn.Variable([1, 5], need_grad=True)
        x2 = nn.Variable([1, 5], need_grad=True)
        x3 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)
        xx2 = F.identity(x2)
        y1 = F.mul2(xx1, xx2)  # (1)
        xx3 = F.identity(x3)
        y2 = F.add2(xx2, xx3)  # (2)
        y3 = F.add2(y1, y2)  # (3)

        answer = []
        answer.append([False])
        answer.append([False])
        answer.append([False, False])  # (1)
        answer.append([False])
        answer.append([False, True])  # (2) use xx2 in backward
        answer.append([True, True])  # (3)

        y3.forward(clear_no_need_grad=True)
        self.check_input_data_clear_called_flags(answer)

    # Test for inplace after branching.
    def test_clear_input_if_no_need_grad_branch2(self):
        x1 = nn.Variable([1, 5], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)
        y2 = F.add_scalar(y1, inplace=True)
        z1 = F.add_scalar(xx1)
        z2 = F.add_scalar(z1)
        y3 = F.add2(y2, z2)

        answer = []
        answer.append([False])
        answer.append([False])
        answer.append([False])
        answer.append([True])
        answer.append([True])
        answer.append([False, True])

        y3.forward(clear_no_need_grad=True)
        self.check_input_data_clear_called_flags(answer)

    # Test for only clearing bias in convolution.
    def test_clear_input_if_no_need_grad_convolution(self):
        x1 = nn.Variable([1, 1, 2], need_grad=True)
        x2 = nn.Variable([1, 1, 2], need_grad=True)
        x3 = nn.Variable([1], need_grad=True)

        inp = F.identity(x1)
        weight = F.identity(x2)
        bias = F.identity(x3)
        y = F.convolution(inp, weight, bias)  # (1)

        answer = []
        answer.append([False])
        answer.append([False])
        answer.append([False])
        answer.append([False, False, True])  # (1) clears bias

        y.forward(clear_no_need_grad=True)
        self.check_input_data_clear_called_flags(answer)

    # Test for only clearing beta in batch_normalization.
    @pytest.mark.parametrize("batch_stat", [False, True])
    def test_clear_input_if_no_need_grad_batch_normalization(self, batch_stat):
        x1 = nn.Variable([1, 1, 2], need_grad=True)
        x2 = nn.Variable([1, 1, 1], need_grad=True)
        x3 = nn.Variable([1, 1, 1], need_grad=True)
        x4 = nn.Variable([1, 1, 1], need_grad=True)
        x5 = nn.Variable([1, 1, 1], need_grad=True)

        x = F.identity(x1)
        beta = F.identity(x2)
        gamma = F.identity(x3)
        if batch_stat:
            y = F.batch_normalization(
                x, beta, gamma, x4, x5, batch_stat=batch_stat)
        else:
            mean = F.identity(x4)
            var = F.identity(x5)
            y = F.batch_normalization(
                x, beta, gamma, mean, var, batch_stat=batch_stat)

        answer = []
        answer.append([False])
        answer.append([False])
        answer.append([False])
        if not batch_stat:
            answer.append([False])
            answer.append([False])
        answer.append([False, True, False, False, False])

        y.forward(clear_no_need_grad=True)
        self.check_input_data_clear_called_flags(answer)


class TestClearOutputGrad():

    def check_grad_cleared_flags(self, answer):
        result = clear_called_flag_recorder.get_output_clear_called_flags()
        assert len(result) == len(answer)
        for i, flags in enumerate(answer):
            assert len(result[i]) == len(flags)
            for j, flag in enumerate(flags):
                assert flag == result[i][j][1]

    def setup_method(self):
        clear_called_flag_recorder.activate_clear_called_flag_recorder()

    def teardown_method(self):
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

    # Test for the type of grad given to backward.
    @pytest.mark.parametrize("grad", [1, None, np.ndarray([1]), nn.NdArray([1])])
    def test_clear_output_grad_argument(self, grad):
        x1 = nn.Variable([1], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)

        answer_grad = []
        if grad is None or isinstance(grad, nn.NdArray):
            answer_grad.append([False])  # y1
        else:
            answer_grad.append([True])  # y1
        answer_grad.append([True])  # xx1

        y1.forward(clear_no_need_grad=True)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        y1.backward(clear_buffer=True, grad=grad)

        self.check_grad_cleared_flags(answer_grad)
        assert y1.grad.clear_called == False

    # Test for an inplaced variable.
    def test_clear_output_grad_inplace(self):
        x1 = nn.Variable([1], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1, inplace=True)
        y2 = F.add_scalar(y1)

        answer_grad = []
        answer_grad.append([True])
        answer_grad.append([True])
        answer_grad.append([True])

        y2.forward(clear_no_need_grad=True)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        y2.backward(clear_buffer=True)

        self.check_grad_cleared_flags(answer_grad)

    # Test for a variable shared with two layer functions.
    def test_clear_output_grad_shared_variable(self):
        x1 = nn.Variable([1], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)
        y2 = F.add_scalar(xx1)
        y3 = F.add2(y1, y2)

        answer_grad = []
        answer_grad.append([True])
        answer_grad.append([True])
        answer_grad.append([True])
        answer_grad.append([True])

        y3.forward(clear_no_need_grad=True)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        y3.backward(clear_buffer=True)

        self.check_grad_cleared_flags(answer_grad)

    # Test for a persistent variable.
    def test_clear_output_grad_persistent(self):
        x1 = nn.Variable([1], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)
        y2 = F.add_scalar(y1)

        xx1.persistent = True
        y2.persistent = True

        answer_grad = []
        answer_grad.append([False])  # y2
        answer_grad.append([True])  # y1
        answer_grad.append([False])  # xx1

        y2.forward(clear_no_need_grad=True)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        y2.backward(clear_buffer=True)

        self.check_grad_cleared_flags(answer_grad)

    # Test for the input variables of sink.
    # In the case where Function::prohibit_clear_input_buffers returns true,
    # these inputs must not be cleared from any function.
    def test_clear_output_grad_prohibit_clear_input(self):
        x1 = nn.Variable([1], need_grad=True)

        xx1 = F.identity(x1)
        y1 = F.add_scalar(xx1)
        y2 = F.add_scalar(xx1)
        y3 = F.sink(y1, y2)

        answer_grad = []
        answer_grad.append([True])  # y3
        answer_grad.append([False])  # y2
        answer_grad.append([False])  # y1
        answer_grad.append([True])  # xx1

        y3.forward(clear_no_need_grad=True)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        y3.backward(clear_buffer=True)

        self.check_grad_cleared_flags(answer_grad)
