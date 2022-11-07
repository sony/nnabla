# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
import copy
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.testing import assert_allclose, clear_called_flag_recorder
from nbla_test_utils import list_context, quit_with_gc
from nnabla.function import PythonFunction


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


class FuncForTestClearBuffersInAutoForward(PythonFunction):
    ''' A Function takes two inputs and two outputs, where one of the inputs
        and one of the outputs have the grad dependency. 
    '''
    @property
    def name(self):
        return self.__class__.__name__

    def min_outputs(self):
        return 2

    def setup_impl(self, inputs, outputs):
        outputs[0].reset_shape(inputs[0].shape, True)
        outputs[1].reset_shape(inputs[1].shape, True)

    def forward_impl(self, inputs, outputs):
        # Imperative call
        outputs[0].data.copy_from(F.identity(inputs[0].data))
        outputs[1].data.copy_from(F.identity(inputs[1].data))

    def grad_depends_input_data(self, i, j):
        # Only the inputs[0] has the grad dependency.
        if j == 0:
            return True
        return False

    def grad_depends_output_data(self, i, o):
        # Only the outputs[0] has the grad dependency.
        if o == 0:
            return True
        return False

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        # Nothing to do
        pass


def func_for_test_clear_buffer_in_auto_forward(x0, x1, ctx=None):
    return FuncForTestClearBuffersInAutoForward(ctx)(x0, x1)


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("unlinked_variable", [True, False])
@pytest.mark.parametrize("need_grad", [True, False])
def test_clear_buffers_in_auto_forward(persistent, unlinked_variable, need_grad):
    @quit_with_gc
    def local_scope(x0, x1):
        # Input layer
        h0 = F.identity(x0)
        h1 = F.identity(x1)

        # Clear the grad-dependent Variable h0 when the layer functions
        # associating with h0 have need_grad=False.
        h0.need_grad = need_grad
        h1.need_grad = need_grad

        # Intermediate layer
        h2, h3 = func_for_test_clear_buffer_in_auto_forward(h0, h1)

        # Do not clear the persistent Variables
        h1.persistent = persistent
        h3.persistent = persistent

        # The stand-alone unlinked variable is not cleared because it is dealt
        # with the input of a graph consisting of the single node.
        if unlinked_variable:
            unlinked_h0 = h0.get_unlinked_variable()
            unlinked_h1 = h1.get_unlinked_variable()
            unlinked_h2 = h2.get_unlinked_variable()
            unlinked_h3 = h3.get_unlinked_variable()

        # Output layer
        # Returning the NdArrays to check their memory clearance.
        ret = (h2 + h3, h0.data, h1.data, h2.data, h3.data)

        if unlinked_variable:
            # Ensure that unlinked variables deallocated after h0-3.
            del h0
            del h1
            del h2
            del h3
        return ret

    # Test
    shape_input = [2, 3, 4]
    x0 = nn.Variable(shape_input, need_grad=True)
    x1 = nn.Variable(shape_input, need_grad=True)

    with nn.auto_forward():
        # Test to clear the variables in a local scope after the exit.
        # h0, h1, h2, and h3 are expected to be deleted conditionally.
        y, h0_ndarr, h1_ndarr, h2_ndarr, h3_ndarr = local_scope(x0, x1)

        # Test to clear the variable after reassignment.
        # y is expected to be deleted conditionally.
        y.persistent = persistent
        if unlinked_variable:
            y_unlinked = y.get_unlinked_variable()
        y_ndarr = y.data
        y = F.identity(y)

    assert h0_ndarr.clear_called == (
        (not need_grad) and (not unlinked_variable))
    assert h1_ndarr.clear_called == (
        (not persistent) and (not unlinked_variable))
    assert h2_ndarr.clear_called == False
    assert h3_ndarr.clear_called == (
        (not persistent) and (not unlinked_variable))
    assert y_ndarr.clear_called == (
        (not persistent) and (not unlinked_variable))


def test_clear_buffers_in_auto_forward_with_narrow():

    def model(x, fix_parameters=True):
        h = PF.affine(x, 2, fix_parameters=fix_parameters, name='fc1')
        h2 = PF.affine(h, 2, fix_parameters=fix_parameters, name='fc2')
        return h2

    shape_input = [2, 3]
    x = nn.Variable(shape_input, need_grad=False)

    # Create parameters
    model(x)

    # Pack all parameters and gradients.
    # Making a local function to make sure that intermediate Python variables
    # are deleted
    def pack_parameters(params):
        # Get size list
        size_list = []
        for k, v in params.items():
            size_list.append(v.size)
        size = sum(size_list)

        # Create a packed NdArray for parameters.
        weights = nn.NdArray([size])

        # Set narrowed arrays to parameters.
        start = 0
        for param, sz in zip(params.values(), size_list):
            w = weights.narrow(0, start, sz)
            param.data = w.view(param.shape)
            start += sz

        return weights

    params = nn.get_parameters()
    weights = pack_parameters(params)

    x.d = 0.1
    with nn.auto_forward():
        y = model(x, fix_parameters=True)
        # Remove all Python references of narrowed arrays
        # Even if all references are deletec, we expect the values are not
        # cleared
        nn.clear_parameters()
        del weights

    def check_params_not_cleared(f):
        if not f.name.startswith('Affine'):
            return
        for inp in f.inputs[1:]:
            assert not inp.data.clear_called

    y.visit(check_params_not_cleared)

    # Check exceptions raised during __dealloc__.
    nn.Variable._check_exception_at_dealloc()

    # People never do this, but testing if exception is properly raised.
    with nn.auto_forward():
        h = x + 1
        base = nn.NdArray([2 * h.size])
        narrowed = base.narrow(0, 0, h.size)
        h.data = narrowed.view(h.shape)
        h = h + 1
    with pytest.raises(RuntimeError):
        # Check exceptions raised during __dealloc__.
        nn.Variable._check_exception_at_dealloc()


def test_python_user_reference_counts():
    shape = [2, 3, 4]

    # Create a Variable
    x0 = nn.Variable(shape)
    assert x0.get_number_of_references == 1

    # Shallow copy does not change the reference counts.
    x0_shallow_copy = x0
    assert x0.get_number_of_references == 1
    assert x0_shallow_copy.get_number_of_references == 1

    del x0_shallow_copy
    assert x0.get_number_of_references == 1

    # Getting numpy array does not change the reference counts
    d = x0.d
    assert x0.get_number_of_references == 1

    # Getting NdArray does not change the reference counts. NdArray is not
    # protected by automatic memory clearing both of this reference counting
    # in dynamic-graph mode and of the graph engine in static-graph mode.
    x0_data = x0.data
    assert x0.get_number_of_references == 1

    # Simple Function call does not change the reference counts.
    y0 = F.identity(x0)
    assert x0.get_number_of_references == 1

    # Inplace Function call increases the reference counts. In this case,
    # x0 and y0 share the same memory object via SyncedArray level.
    y0 = F.reshape(x0, shape)
    assert x0.get_number_of_references == 2
    assert y0.get_number_of_references == 2

    # Sharing NdArray increases the reference counts. In this case, x0 and x1
    # x0 and y0 share the same memory object via NdArray level.
    x1 = nn.Variable(shape)
    assert x0.get_number_of_references == 2
    assert y0.get_number_of_references == 2
    assert x1.get_number_of_references == 1
    x1.data = x0_data
    assert x0.get_number_of_references == 3
    assert y0.get_number_of_references == 3
    assert x1.get_number_of_references == 3

    # Sharing Variable increases the reference counts. In this case, y2 and x3
    # share the same memory object via Variable level.
    x2 = nn.Variable(shape)
    x3 = nn.Variable(shape)
    assert x2.get_number_of_references == 1
    assert x3.get_number_of_references == 1
    y2 = F.identity(x2)
    x3.rewire_on(y2)
    assert x2.get_number_of_references == 1
    assert y2.get_number_of_references == 2
    assert x3.get_number_of_references == 2

    # Another way to share Variable. In this case, x4 and x5 share the same
    # memory object via Variable level.
    x4 = nn.Variable(shape)
    assert x4.get_number_of_references == 1
    x5 = x4.get_unlinked_variable()
    assert x4.get_number_of_references == 2
    assert x5.get_number_of_references == 2

    # Note: The case of shareing CgVariable is not happen in the NNabla version
    #       when this test is implemented because CgVariable and Varable are
    #       are identical.

    # Local scope changes the reference counts locally.
    @quit_with_gc
    def local_scope(x6):
        assert x6.get_number_of_references == 1
        x7 = x6.get_unlinked_variable()
        assert x6.get_number_of_references == 2
        assert x7.get_number_of_references == 2

    x6 = nn.Variable(shape, need_grad=True)
    assert x6.get_number_of_references == 1
    local_scope(x6)
    assert x6.get_number_of_references == 1


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


class TestRecomputation():
    def teardown_method(self):
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

    def check_input_data_clear_called_flags(self, answer):
        result = clear_called_flag_recorder.get_input_clear_called_flags()
        assert len(result) == len(answer)
        for i, flags in enumerate(answer):
            assert len(result[i]) == len(flags)
            for j, flag in enumerate(flags):
                assert flag == result[i][j][0]

    def check_recomputation(self, seed, graph, inputs):
        def forward_backward_and_get_grads(y):
            # Initialize grads
            for input in inputs:
                if input.need_grad:
                    input.grad.zero()

            y.forward(clear_no_need_grad=True)
            y.backward(clear_buffer=True)

            # Get grads
            grads = []
            for input in inputs:
                if input.need_grad:
                    grads.append(copy.deepcopy(input.g))

            return grads

        # Set random input data.
        rng = np.random.RandomState(seed)
        for input in inputs:
            input.d = rng.randn(*input.shape)

        # Calculate reference grads.
        y_ref = graph(*inputs)
        # Disable recompute flags for generating reference grads.

        def disable_recompute_flag(f):
            for input in f.inputs:
                input.apply(recompute=False)
        y_ref.visit(disable_recompute_flag)
        grads_expected = forward_backward_and_get_grads(y_ref)

        y = graph(*inputs)
        grads_actual = forward_backward_and_get_grads(y)
        for a, e in zip(grads_actual, grads_expected):
            assert_allclose(a, e, rtol=0, atol=0)

    # Check setting up recompute flag.
    def test_recompute_flag(self):
        x0 = nn.Variable((1, 1), need_grad=True)
        x1 = F.sin(x0).apply(recompute=True)
        x2 = F.sin(x1).apply(recompute=False)
        x3 = F.sin(x2)

        assert x0.recompute == False
        assert x1.recompute == True
        assert x2.recompute == False
        assert x3.recompute == False

    # Check whether input data is cleared when recompute flag is True.
    def test_clear_input_data(self):
        x0 = nn.Variable((1, 1), need_grad=True)
        # `F.sin` input data is always needed for grad calculation
        x1 = F.sin(x0).apply(recompute=True)
        x2 = F.sin(x1).apply(recompute=False)
        x3 = F.sin(x2)

        answer = []
        answer.append([False])  # x0
        answer.append([True])  # x1
        answer.append([False])  # x2

        clear_called_flag_recorder.activate_clear_called_flag_recorder()

        x3.forward(clear_no_need_grad=True)
        self.check_input_data_clear_called_flags(answer)

        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

    # Check claering output which needs `setup_recompute` for recomputation.
    def test_clearing_without_recompute_flag(self):
        x0 = nn.Variable((1, 128, 128), need_grad=True)
        x1 = F.sin(x0).apply(recompute=True)
        x2 = F.dropout(x1)
        x3 = F.sin(x2).apply(recompute=True)
        x4 = F.sin(x3).apply(recompute=True)
        y = F.identity(x4)

        # Skip this code temporarily since it cause
        # randomly crash when perform CI testing on windows 10 with nnabla-cuda-ext
        pytest.skip(
            'Skipped for randomly crash when perform CI testing on windows 10 with nnabla-cuda-ext')

        y.forward(clear_no_need_grad=True)
        x2.data.clear()
        with pytest.raises(RuntimeError, match="Failed `called_setup_recompute_`"):
            # x2.data cannot be recomputed correctly since `setup_recompute` is not called during forward propagation.
            # Backward should raise when some intermediate variables are cleared by user.
            y.backward()

    # Check recomputed data value.
    @pytest.mark.parametrize("seed", [313])
    def test_recomputed_data_value(self, seed):
        rng = np.random.RandomState(seed)
        a0 = nn.Variable((2, 3), need_grad=True)
        b0 = nn.Variable((2, 3), need_grad=True)
        a0.d = rng.randn(*a0.shape)
        b0.d = rng.randn(*b0.shape)

        a1 = F.sin(a0).apply(recompute=True)
        a2 = F.sin(a1)
        a3 = F.sin(a2)

        b1 = F.sin(b0)
        b2 = F.sin(b1).apply(recompute=True)
        b3 = F.sin(b2)

        c0 = F.mul2(a3, b3).apply(recompute=True)
        c1 = F.sin(c0)

        # Forward

        # Get output data which will be recomputed.
        ref_data = []  # data of a0, b2 and c0 will be stored.

        def get_output_data(nnabla_func):
            outputs = nnabla_func.outputs
            for output in outputs:
                if output.recompute:
                    ref_data.append(copy.deepcopy(output.d))
        c1.forward(function_post_hook=get_output_data)

        # Backward

        # Get recomputed data
        act_data = []

        def get_recomputed_data(nnabla_func):
            inputs = nnabla_func.inputs
            for input in inputs:
                if input.recompute:
                    act_data.append(copy.deepcopy(input.d))
        c1.backward(function_pre_hook=get_recomputed_data)
        # Make the order the same as `ref_data`.
        act_data.reverse()

        # Check recomputed data
        for act, ref in zip(act_data, ref_data):
            assert_allclose(act, ref, rtol=0, atol=0)

    @pytest.mark.parametrize("seed", [313])
    def test_grad_value_simple(self, seed):
        x = nn.Variable((2, 3), need_grad=True)

        inputs = (x,)

        def graph(x):
            y = F.sin(x).apply(recompute=True)
            y = F.cos(y)
            return y

        self.check_recomputation(seed, graph, inputs)

    @pytest.mark.parametrize("seed", [313])
    @pytest.mark.parametrize("need_grad_x1", [False, True])
    @pytest.mark.parametrize("need_grad_x2", [False, True])
    def test_grad_value_with_branch(self, seed, need_grad_x1, need_grad_x2):
        x1 = nn.Variable((2, 3), need_grad=need_grad_x1)
        x2 = nn.Variable((2, 3), need_grad=need_grad_x2)

        inputs = (x1, x2)

        def graph(x1, x2):
            x1 = F.identity(x1).apply(recompute=True)
            x2 = F.identity(x2).apply(recompute=True)
            y = F.mul2(x1, x2)
            y = F.identity(y)
            return y

        self.check_recomputation(seed, graph, inputs)

    # Check `setup_recompute`
    @pytest.mark.parametrize("seed", [313])
    def test_grad_value_with_random_function(self, seed):
        x1 = nn.Variable((2, 3), need_grad=True)

        inputs = (x1,)

        def graph(x1):
            x1 = F.identity(x1).apply(recompute=True)
            x2 = F.randn(shape=x1.shape, seed=123).apply(recompute=True)
            x3 = F.rand(shape=x1.shape, seed=456).apply(recompute=True)
            y = F.mul2(x1, x2).apply(recompute=True)
            y = F.mul2(y, x3).apply(recompute=True)
            y = F.identity(y)
            return y

        self.check_recomputation(seed, graph, inputs)

    @pytest.mark.parametrize("seed", [313])
    def test_grad_value_with_output_dependent_function(self, seed):
        """
        Gradient values are tested for the function which depends on output data.
        Here, we test a following case that variable `h` will be recomputed and
        its data is needed for the `F.swish` backward.
        x -> F.swish -> h -> F.interpolate -> y
        """
        def graph(x0):
            # F.swish -> F.interpolate
            x1 = F.swish(x0)
            x1.apply(recompute=True)
            x2 = F.interpolate(x1, scale=(2,))
            return x2

        x = nn.Variable((2, 3), need_grad=True)
        inputs = (x,)
        self.check_recomputation(seed, graph, inputs)

    @pytest.mark.parametrize("seed", [313])
    def test_with_persistent_flag(self, seed):
        x = nn.Variable((2, 3), need_grad=True)

        inputs = (x,)

        def graph(x0):
            x1 = F.sin(x0).apply(recompute=True)
            # Set `recompute` and `persistent` flag at the same time
            x2 = F.sin(x1).apply(recompute=True, persistent=True)
            x3 = F.sin(x2).apply(recompute=True)
            y = F.sin(x3)
            return y

        y = graph(x)

        # Trace data clearing during forward propagation.
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        y.forward(clear_no_need_grad=True)
        expected = [
            [False],  # x0: graph input
            [True],  # x1: Cleared because `recompute=True`
            [False],  # x2: Not cleared because `persistent=True`
            [True],  # x3: Cleared because `recompute=True`
        ]
        self.check_input_data_clear_called_flags(expected)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

        # Check grad value
        self.check_recomputation(seed, graph, inputs)

    @pytest.mark.parametrize("seed", [313])
    def test_with_inplacing(self, seed):
        x = nn.Variable((2, 3), need_grad=True)

        inputs = (x,)

        def graph(x0):
            x1 = F.sin(x0).apply(recompute=True)
            # Set `recompute` flag to the inplaced variable.
            x2 = F.reshape(x1, (3, 2), inplace=True).apply(recompute=True)
            x3 = F.sin(x2).apply(recompute=True)
            y = F.sin(x3)
            return y

        y = graph(x)

        # Trace data clearing during forward propagation.
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        y.forward(clear_no_need_grad=True)
        expected = [
            [False],  # x0: graph input
            [False],  # x1: Not cleared because inplaced data
            [False],  # x2: Not cleared because inplaced data
            [True],  # x3: Cleared because `recompute=True`
        ]
        self.check_input_data_clear_called_flags(expected)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

        # Check grad value
        self.check_recomputation(seed, graph, inputs)

    # Check clear of recomputed data on the subgraph which is not back-propagated.
    def test_clear_data_on_not_bwd_path(self):
        a0 = nn.Variable((2, 3), need_grad=True)
        a1 = F.identity(a0).apply(recompute=True)
        a2 = F.sin(a1).apply(recompute=True)

        # These three variables are not back-propagated.
        b0 = nn.Variable((2, 3), need_grad=False)
        b1 = F.identity(b0).apply(recompute=True)
        b2 = F.sin(b1).apply(recompute=True)

        c1 = F.add2(a2, b2).apply(recompute=True)
        c2 = F.sin(c1)

        # Forward
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        c2.forward(clear_no_need_grad=True)
        # Data which will be recomputed must be cleared during forward propagation.
        expected = [
            [False],  # a0
            [True],  # a1
            [False],  # b0
            [True],  # b1
            [True, True],  # a2, b2
            [True],  # c1
        ]
        self.check_input_data_clear_called_flags(expected)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

        # Backward
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        c2.backward(clear_buffer=True)
        # b1 is not on backward path and must be cleared during recomputation.
        expected = [
            # Recomputation
            [False],  # a0
            [False],  # a1
            [False],  # b0
            [True],  # b1 (not on backward path) must be cleared
            [True, True],  # a2, b2
            [False],  # c1
            # Backward propagation
            [True, True],  # a2, b2
            [False],  # a1
            [False],  # a0
        ]
        self.check_input_data_clear_called_flags(expected)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

    # Check clear of data not need for grad calculation during recomputation.
    def test_clear_no_need_grad_during_recomputation(self):
        x0 = nn.Variable((2, 3), need_grad=True)

        x1 = F.identity(x0).apply(recompute=True)
        # x2.data must be cleared just after recomputation because they are not need for backward propagation.
        x2 = F.sin(x1).apply(recompute=True)
        x3 = F.identity(x2).apply(recompute=True)
        x4 = F.sin(x3)

        # Forward
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        x4.forward(clear_no_need_grad=True)
        # All intermediate data must be cleared.
        expected = [
            [False],  # x0
            [True],  # x1
            [True],  # x2
            [True],  # x3
        ]
        self.check_input_data_clear_called_flags(expected)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

        # Backward
        clear_called_flag_recorder.activate_clear_called_flag_recorder()
        x4.backward(clear_buffer=True)
        expected = [
            # Recomputation
            [False],  # x0
            [False],  # x1
            [True],  # x2: not need for grad calculation
            # Backward propagation
            [False],  # x3
            [True],  # x2
            [False],  # x1
            [False],  # x0
        ]
        self.check_input_data_clear_called_flags(expected)
        clear_called_flag_recorder.deactivate_clear_called_flag_recorder()

    # Check recompute recursion stops at checkpoint.
    def test_checkpoint(self):
        x0 = nn.Variable((2, 3), need_grad=True)

        x1 = F.sin(x0).apply(recompute=True)
        x2 = F.sin(x1).apply(recompute=True)
        x3 = F.sin(x2)  # Checkpoint 1 (recompute == False)
        x4 = F.sin(x3).apply(recompute=True)
        x5 = F.sin(x4).apply(recompute=True)
        x6 = F.sin(x5)  # Checkpoint 2 (recompute == False)
        x7 = F.sin(x6).apply(recompute=True)
        x8 = F.sin(x7).apply(recompute=True)

        # All intermediate data except checkpoints will be cleared during forward propagation.
        x8.forward(clear_no_need_grad=True)

        # Trace clear_called flags of `x2` and `x5` during backward propagation.
        # clear_called flag changes True to False when the data is recomputed.
        act_flags = []

        def get_clear_called_flags(nnabla_func):
            act_flags.append([x2.data.clear_called, x5.data.clear_called])
        x8.backward(function_post_hook=get_clear_called_flags)
        ref_flags = [
                     # [x2, x5] clear_called flags
                     [True, True],  # After F.sin(x7) backward
                     [True, True],  # After F.sin(x6) backward
                     [True, False],  # After F.sin(x5) backward
                     [True, False],  # After F.sin(x4) backward
                     [True, False],  # After F.sin(x3) backward
                     [False, False],  # After F.sin(x2) backward
                     [False, False],  # After F.sin(x1) backward
                     [False, False],  # After F.sin(x0) backward
                    ]

        assert (ref_flags == act_flags)

    # Test unnecessary recomputation with single recomputation recursion.
    def test_unnecessary_traverse_0(self):
        # No need grad path
        a0 = nn.Variable((2, 3), need_grad=False)
        a1 = F.sin(a0).apply(recompute=True)
        # Need grad path
        b0 = nn.Variable((2, 3), need_grad=True)
        b1 = F.sin(b0).apply(recompute=True)
        # branch
        c = F.add2(a1, b1)

        # Check whether unnecessary recomputation for `a1.data` is performed.

        c.forward(clear_no_need_grad=True)
        assert (a1.data.clear_called == True)
        assert (b1.data.clear_called == True)

        # Exec backward without clearing buffer to check whether recomputation is performed by seeing `clear_called` flag.
        c.backward(clear_buffer=False)
        # a1.data is still cleared. (Recalculation is not performed)
        assert (a1.data.clear_called == True)
        # b1.data is set. (Recalculation is performed)
        assert (b1.data.clear_called == False)

    # Test recomputation recursion depth.
    def test_unnecessary_traverse_1(self):
        a0 = nn.Variable((2, 3), need_grad=False)
        # `a1` will not be recomputed since `a2` will not be cleared.
        a1 = F.sin(a0).apply(recompute=True)
        a2 = F.cos(a1)
        a3 = F.sin(a2).apply(recompute=True)  # 'a3` will be recomputed.

        b0 = nn.Variable((2, 3), need_grad=True).apply(recompute=True)
        b1 = F.identity(b0).apply(recompute=True)

        c = F.mul2(a3, b1).apply(recompute=True)

        # Check recomputation recursion stops when `a3.data` is calculated.

        c.forward(clear_buffer=False)
        # `a1.data` is cleared because `recompute` flag is `true`.
        assert (a1.data.clear_called == True)
        # `a2.data` is not cleared because `recompute` flag is `false`.
        assert (a2.data.clear_called == False)
        c.backward(clear_buffer=False)
        # If the recursive call reached to `a1`, `a1.data` should be set by recomputation.
        # However, the recursive call stops at `a2` whose data is not cleared.
        assert (a1.data.clear_called == True)

    # Test unnecessary recomputation for whole graph.
    def test_unnecessary_traverse_2(self):
        def fail_with_not_cleared_data(nnabla_func):
            inputs = nnabla_func.inputs
            for input in inputs:
                if input.parent is None:
                    continue
                if not input.data.clear_called:
                    # Not cleared (recomputed) data is found.
                    pytest.fail()

        # Prepare graph does not need any recomputation.
        x1 = nn.Variable((2, 3), need_grad=True)
        x1 = F.identity(x1).apply(recompute=True)
        x2 = nn.Variable((2, 3), need_grad=True)
        x2 = F.identity(x2).apply(recompute=True)
        y = F.add2(x1, x2).apply(recompute=True)
        y = F.identity(y).apply(recompute=True)

        # Check unnecessary recomputation.
        y.forward(clear_no_need_grad=True)
        y.backward(function_pre_hook=fail_with_not_cleared_data)

    @pytest.mark.parametrize("recompute_flag", [False, True])
    def test_with_statement_variable_creation(self, recompute_flag):
        """
        Test for setting recompute flags with Python `with` statement.
        """

        # Create a new Variable
        x1 = nn.Variable((2, 3))
        assert x1.recompute == False

        with nn.recompute(recompute_flag):
            # Create Variable by `__cinit__()`
            y1 = nn.Variable((2, 3))
            assert y1.recompute == recompute_flag

            # Create Variable by `create_from_cvariable()`
            y2 = x1.reshape((3, 2), unlink=True)
            assert y2.recompute == recompute_flag

            # Create Variable by `create_from_cg_variable()`
            y3 = F.relu(x1)
            assert y3.recompute == recompute_flag

            # Create Variable by `from_numpy_array()`
            data = np.array((2, 3))
            y4 = nn.Variable.from_numpy_array(data)
            assert y4.recompute == recompute_flag

            # Create Variable by `get_unlinked_variable()`
            y5 = x1.get_unlinked_variable()
            assert y5.recompute == recompute_flag

            # Recompute flag for referenced Variable must not be overwritten.
            # More detail tests are performed by `test_nested_with_statement`
            y6 = x1
            assert y6.recompute == False

            # Direct function connection
            y7 = F.relu(F.relu(x1))

        # Create a new Variable after with statement
        x2 = nn.Variable((2, 3))
        assert x2.recompute == False

        # Check recompute flag of forcibly got Pyhon Variable.
        assert y7.parent.inputs[0].recompute == recompute_flag

        # Check default recompute flag for nn.recompute()
        with nn.recompute():
            x = nn.Variable((2, 3))
            assert x.recompute == True

    # Recompute flag for first nest
    @pytest.mark.parametrize("f1", [False, True])
    # Recompute flag for second nest
    @pytest.mark.parametrize("f2", [False, True])
    # Recompute flag for third nest
    @pytest.mark.parametrize("f3", [False, True])
    def test_nested_with_statement(self, f1, f2, f3):
        """
        Test for nested Pyhon `with` statement of recomputation.
        """

        x0 = nn.Variable((2, 3))
        assert x0.recompute == False

        # Nest 1
        with nn.recompute(f1):
            x1 = nn.Variable((2, 3))
            x0_1 = x0
            assert x1.recompute == f1
            assert x0_1.recompute == False

            # Nest 2
            with nn.recompute(f2):
                x2 = nn.Variable((2, 3))
                x0_2 = x0
                x1_2 = x1
                assert x2.recompute == f2
                assert x0_2.recompute == False
                assert x1_2.recompute == f1

                # Nest 3
                with nn.recompute(f3):
                    x3 = nn.Variable((2, 3))
                    x0_3 = x0
                    x1_3 = x1
                    x2_3 = x2
                    assert x3.recompute == f3
                    assert x0_3.recompute == False
                    assert x1_3.recompute == f1
                    assert x2_3.recompute == f2

                x2 = nn.Variable((2, 3))
                x0_2 = x0
                x1_2 = x1
                assert x2.recompute == f2
                assert x0_2.recompute == False
                assert x1_2.recompute == f1

            x1 = nn.Variable((2, 3))
            x0_1 = x0
            assert x1.recompute == f1
            assert x0_1.recompute == False

        x0 = nn.Variable((2, 3))
        assert x0.recompute == False

    # Recompute flag for first `with` block
    @pytest.mark.parametrize("f1", [False, True])
    # Recompute flag for second `with` block
    @pytest.mark.parametrize("f2", [False, True])
    def test_sequential_with_statement(self, f1, f2):
        """
        Test for sequential use of with statement.
        """
        x = nn.Variable((2, 3))
        assert x.recompute == False

        # First `with` block
        with nn.recompute(f1):
            y = F.relu(x)
            assert y.recompute == f1
            y = F.sin(y)
            assert y.recompute == f1

        assert y.recompute == f1

        y = F.relu(y)
        assert y.recompute == False

        # Second `with` block
        with nn.recompute(f2):
            y = F.relu(x)
            assert y.recompute == f2
            y = F.sin(y)
            assert y.recompute == f2

        assert y.recompute == f2

        y = F.relu(y)
        assert y.recompute == False

    @pytest.mark.parametrize("recompute_flag", [False, True])
    def test_recompute_fn_decorator(self, recompute_flag):
        """
        Test for setting recompute flags with function decorator `nn.recompute_fn()`.
        """

        # Specifying recompute flag
        @nn.recompute_fn(recompute_flag)
        def func2(x):
            assert x.recompute == False
            y = F.relu(x)
            assert y.recompute == recompute_flag
            return y

        # Check recompute flags
        x2 = nn.Variable((2, 3))
        assert x2.recompute == False
        y2 = func2(x2)
        assert y2.recompute == recompute_flag

    def test_recompute_fn_decorator_default_use(self):
        """
        Test for setting recompute flags with function decorator `nn.recompute_fn()` without specifying recompute flag.
        """

        # Default usage
        @nn.recompute_fn()
        def func1(x):
            assert x.recompute == False
            y = F.relu(x)
            assert y.recompute == True
            return y

        # Check recompute flags
        x1 = nn.Variable((2, 3))
        assert x1.recompute == False
        y1 = func1(x1)
        assert y1.recompute == True

    @pytest.mark.parametrize("recompute_flag", [False, True])
    def test_recompute_fn_decorator_multiple_inputs_outputs(self, recompute_flag):
        """
        Test for the use of `nn.recompute_fn()` with a function which have multiple inputs, outpus, args and kwargs.
        """

        # Define sample function with multiple inputs and outputs
        @nn.recompute_fn(recompute_flag)
        def func(x1, x2, val, axis, reverse=False, alpha=0.2):
            # Check args and kwargs passed correctly
            assert val == 3.14
            assert axis == 0
            assert reverse == True
            assert alpha == 0.3

            y1 = F.cumsum(x1, axis, reverse=reverse)
            y2 = x2 * val

            y3 = y1 + y2
            y3 = F.leaky_relu(y3, alpha=alpha)

            # Check recompute flags for variables defined inside this function
            assert y1.recompute == recompute_flag
            assert y2.recompute == recompute_flag
            assert y3.recompute == recompute_flag

            return y2, y3

        x1 = nn.Variable((2, 3))
        x2 = nn.Variable((2, 3))

        y1, y2 = func(x1, x2, 3.14, 0, alpha=0.3, reverse=True)
        assert y1.recompute == recompute_flag
        assert y2.recompute == recompute_flag

    # Recompute flag for outer function
    @pytest.mark.parametrize("f0", [False, True])
    # Recompute flag for first inner function
    @pytest.mark.parametrize("f1", [False, True])
    # Recompute flag for second inner function
    @pytest.mark.parametrize("f2", [False, True])
    def test_nested_recompute_fn_decorator(self, f0, f1, f2):
        """
        Test for setting recompute flags with nested function decorator `nn.recompute_fn()`.
        """

        # First sub function
        @nn.recompute_fn(f1)
        def func1(x):
            assert x.recompute == f0
            y = F.relu(x)
            assert y.recompute == f1
            return y

        # Second sub function
        @nn.recompute_fn(f2)
        def func2(x):
            assert x.recompute == f0
            y = F.sin(x)
            assert y.recompute == f2
            return y

        # Main function
        @nn.recompute_fn(f0)
        def func0(x):
            assert x.recompute == False
            y = F.identity(x)
            assert y.recompute == f0

            # First inner function call
            y = func1(y)
            assert y.recompute == f1

            y = F.relu(y)
            assert y.recompute == f0

            # Second inner function call
            y = func2(y)
            assert y.recompute == f2

            y = F.identity(y)
            assert y.recompute == f0
            return y

        # Call main function
        x = nn.Variable((2, 3))
        y = func0(x)
        assert y.recompute == f0


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("func, num_inputs", [
    (F.relu, 1),
    (F.leaky_relu, 1),
    (F.random_erase, 1),
    (F.add2, 2),
    (F.bc_add2, 2),
    (F.sub2, 2),
    (F.add_scalar, 1),
    (F.mul_scalar, 1),
])
def test_obsolete_inplace_option(inplace, func, num_inputs):
    '''
    This test confirms the construction of graph.
    Since F.log_softmax requires output for backward calculation, graph cannot be constructed if it is inplaced.
    '''
    x0 = nn.Variable((2, 3, 4, 5), need_grad=True)
    x1 = nn.Variable((2, 3, 4, 5), need_grad=True)

    if num_inputs == 1:
        y = F.identity(x0)
        y = F.log_softmax(y)
        y = func(y, inplace=inplace)
        y.forward()
        y.backward()

    elif num_inputs == 2:
        y0 = F.identity(x0)
        y1 = F.identity(x1)
        y0 = F.log_softmax(y0)
        y1 = F.log_softmax(y1)
        y = func(y0, y1, inplace=inplace)
        y.forward()
        y.backward()
