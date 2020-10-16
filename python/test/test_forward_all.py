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
from nnabla.testing import assert_allclose


def initialize_grad(parameters):
    for param in parameters.values():
        param.grad.zero()


@pytest.mark.parametrize("seed", [313])
def test_graph_logreg(seed):
    rng = np.random.RandomState(seed)
    x = nn.Variable([2, 3, 4], need_grad=True)
    w1 = nn.Variable([12, 5], need_grad=True)
    w2 = nn.Variable([12, 5], need_grad=True)
    b1 = nn.Variable([5], need_grad=True)
    b2 = nn.Variable([5], need_grad=True)
    t = nn.Variable([2, 1])
    x.d = rng.randn(*x.shape)
    w1.d = rng.randn(*w1.shape)
    w2.d = rng.randn(*w2.shape)
    b1.d = rng.randn(*b1.shape)
    b2.d = rng.randn(*b2.shape)
    t.d = rng.randint(0, 5, size=t.shape)

    nn.set_default_context(nn.Context())

    # Forwardprop by definintion
    z1 = F.affine(x, w1, b1, 1)
    z2 = F.affine(x, w2, b2, 1)
    l1 = F.softmax_cross_entropy(z1, t, 1)
    L1 = F.mean(l1)
    l2 = F.softmax_cross_entropy(z2, t, 1)
    L2 = F.mean(l2)
    nn.forward_all([L1, L2])

    # Backprop for z1
    # Diff should be initialized since they are always accumulated
    x.g = 0
    w1.g = 0
    b1.g = 0
    L1.backward(clear_buffer=True)

    inputs = [x, w1, b1]

    from nbla_test_utils import \
        compute_analytical_and_numerical_grad_graph as grads
    agrad, ngrad = grads(L1, inputs, 1e-3, False)
    assert_allclose(ngrad, agrad, atol=1e-2)

    # Backprop for z2
    # Diff should be initialized since they are always accumulated
    x.g = 0
    w2.g = 0
    b2.g = 0
    L2.backward(clear_buffer=True)

    inputs = [x, w2, b2]

    from nbla_test_utils import \
        compute_analytical_and_numerical_grad_graph as grads
    agrad, ngrad = grads(L2, inputs, 1e-3, False)
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

    # Forwardprop by definintion
    nn.clear_parameters()
    if model == "mlp":
        with nn.parameter_scope('fc1'):
            z = PF.affine(x, 3)
        z2 = F.relu(z, inplace=True)
        with nn.parameter_scope('fc2'):
            z3 = PF.affine(z2, 5)
            z4 = PF.affine(z2, 5)
    elif model == "recurrent":
        with nn.parameter_scope('fc1'):
            z = PF.affine(x, 4)
            z2 = F.relu(z, inplace=True)
        h = z2
        for _ in range(2):
            with nn.parameter_scope('fc2'):
                h = PF.affine(h, 4)
                h = F.relu(h, inplace=True)
        with nn.parameter_scope('fc3'):
            z3 = PF.affine(h, 5)
            z4 = PF.affine(h, 5)
    elif model == "convolution":
        with nn.parameter_scope('conv1'):
            z = PF.convolution(x, 3, (2, 2))
            z2 = F.relu(z, inplace=True)
        with nn.parameter_scope('fc2'):
            z3 = PF.affine(z2, 5)
            z4 = PF.affine(z2, 5)
    else:
        raise ValueError()
    l1 = F.softmax_cross_entropy(z3, t, 1)
    L1 = F.mean(l1)
    l2 = F.softmax_cross_entropy(z4, t, 1)
    L2 = F.mean(l2)

    # Forwardprop
    nn.forward_all([L1, L2])

    parameters = nn.get_parameters()

    # Backprop for L1
    # Diff should be initialized since they are always accumulated
    x.grad.zero()
    initialize_grad(parameters)
    L1.backward(clear_buffer=True)
    inputs = [x] + list(parameters.values())

    from nbla_test_utils import \
        compute_analytical_and_numerical_grad_graph as grads
    agrad, ngrad = grads(L1, inputs, 1e-3, False)
    assert_allclose(ngrad, agrad, atol=1.05e-2)

    # Backprop for L2
    # Diff should be initialized since they are always accumulated
    x.grad.zero()
    initialize_grad(parameters)
    L2.backward(clear_buffer=True)
    inputs = [x] + list(parameters.values())

    from nbla_test_utils import \
        compute_analytical_and_numerical_grad_graph as grads
    agrad, ngrad = grads(L2, inputs, 1e-3, False)
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
    with nn.parameter_scope("fc0"):
        h0 = PF.affine(x0, 2)
        h0.need_grad = False
    with nn.parameter_scope("fc1"):
        h1 = PF.affine(x1, 2)
    h = h0 + h1
    with nn.parameter_scope("fc"):
        y1 = PF.affine(h, 1)
        y2 = PF.affine(h, 1)
    nn.forward_all([y1, y2])

    y1.backward(clear_buffer=True)
    assert np.all(x0.g == 0)
    assert not np.all(x1.g == 0)

    y2.backward(clear_buffer=True)
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
        z4 = PF.affine(z2, 5)
    l1 = F.softmax_cross_entropy(z3, t, 1)
    L1 = F.mean(l1)
    l2 = F.softmax_cross_entropy(z4, t, 1)
    L2 = F.mean(l2)

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
            nn.forward_all([L1, L2], clear_no_need_grad=cnng)

            # for now, the first backward cannot be
            # called with clear_buffer=True
            L1.backward(clear_buffer=False)
            L2.backward(clear_buffer=cb)
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
def test_graph_forward_clear_buffer(seed, clear_buffer):
    nn.clear_parameters()

    x = nn.Variable((2, 10))
    h = PF.affine(x, 10, name='hidden')
    y1 = PF.affine(h, 10, name='out1')
    y2 = PF.affine(h, 10, name='out2')

    # input
    rng = np.random.RandomState(seed)
    data = rng.randn(*x.shape)

    # reference values
    x.d = data
    y1.forward()
    y2.forward()
    ref_y1 = y1.d.copy()
    ref_y2 = y2.d.copy()

    # check
    nn.forward_all([y1, y2], clear_buffer=clear_buffer)
    assert_allclose(y1.d, ref_y1)
    assert_allclose(y2.d, ref_y2)


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
    yb1 = mlp2(xb, 'b1')
    yb2 = mlp2(xb, 'b2')

    # C. Create directly connected graph.
    xc = nn.Variable((2, 10))
    h = mlp2(xc, 'a')
    yc1 = mlp2(h, 'b1')
    yc2 = mlp2(h, 'b2')

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
    nn.forward_all([yb1, yb2, yc1, yc2], clear_no_need_grad=clear_buffer)
    assert_allclose(yb1.d, yc1.d)
    assert_allclose(yb2.d, yc2.d)

    # Checking backward for yb1 and yc1
    # for now, the first backward cannot be called with clear_buffer=True
    zero_grad()
    yb1.backward(clear_buffer=False)
    gb = backup_params()
    zero_grad()
    yc1.backward(clear_buffer=False)
    gc = backup_params()
    assert_allclose(xa.d, xc.d)
    for b, c in zip(gb, gc):
        assert_allclose(b, c)

    # Checking backward for yb2 and yc2
    zero_grad()
    yb2.backward(clear_buffer=clear_buffer)
    gb = backup_params()
    zero_grad()
    yc2.backward(clear_buffer=clear_buffer)
    gc = backup_params()
    assert_allclose(xa.d, xc.d)
    for b, c in zip(gb, gc):
        assert_allclose(b, c)


@pytest.mark.parametrize("clear_buffer, clear_no_need_grad", [
    (False, False), (True, False), (False, True),
])
def test_intermediate_outputs(clear_buffer, clear_no_need_grad):
    rng = np.random.RandomState(311)

    # unuse cached array to clear buffers immediately
    nn.prefer_cached_array(False)

    x = nn.Variable.from_numpy_array(rng.randn(2, 10))

    h1 = x + 1
    y1 = h1 + 1

    h2 = x + 1
    h2.persistent = True
    y2 = h2 + 1

    nn.forward_all([h1, y1], clear_buffer=clear_buffer,
                   clear_no_need_grad=clear_no_need_grad)
    nn.forward_all([h2, y2], clear_buffer=clear_buffer,
                   clear_no_need_grad=clear_no_need_grad)

    assert_allclose(h1.d, h2.d)
    assert_allclose(y1.d, y2.d)

    # revert perference (this is also done in conftest.py, but just in case)
    nn.prefer_cached_array(True)
