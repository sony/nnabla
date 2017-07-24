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

    # Forwardprop by definintion
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
    assert np.allclose(ngrad, agrad, atol=1e-2)


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
    elif model == "recurrent":
        with nn.parameter_scope('fc1'):
            z = PF.affine(x, 3)
            z2 = F.relu(z, inplace=True)
        h = z2
        for _ in range(2):
            with nn.parameter_scope('fc2'):
                h = PF.affine(h, 3)
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
    assert np.allclose(ngrad, agrad, atol=1.05e-2)


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
                assert np.all(g == g2)
