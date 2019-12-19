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
from nnabla.ext_utils import get_extension_context
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

# Proxy to get the appropriate context
ctx_list = [ctx_fname[0] for ctx_fname in list_context('Convolution')]


def SmallResNet(x, test=False, shared=False):
    h = x

    def conv(x, maps=8, name="conv"):
        h = x
        with nn.parameter_scope(name):
            h = PF.convolution(h, maps, (3, 3), (1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test)
        with nn.parameter_scope("{}-shortcut".format(name)):
            s = PF.convolution(h, maps, (3, 3), (1, 1), with_bias=False)
            h = PF.batch_normalization(h, batch_stat=not test)
        return F.relu(h + s)
    h = conv(h, maps=4, name="conv1")
    h = F.max_pooling(h, (2, 2))
    h = conv(h, maps=4, name="conv2")
    h = conv(h, maps=8, name="conv3") if not shared else conv(
        h, maps=4, name="conv2")
    h = F.average_pooling(h, h.shape[2:])
    h = PF.affine(h, 10)
    return h


@pytest.mark.parametrize("seed", [311])
@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("auto_forward", [True, False])
@pytest.mark.parametrize("flag_grad_outputs", [True, False])
@pytest.mark.parametrize("shared", [False, True])
def test_resnet_expansion(seed, ctx, auto_forward, flag_grad_outputs, shared):
    nn.clear_parameters()

    # Settings
    nn.set_default_context(ctx)
    nn.set_auto_forward(auto_forward)
    b, c, h, w = 4, 3, 32, 32
    n_cls = 10
    rng = np.random.RandomState(seed)

    # Network
    x = nn.Variable.from_numpy_array(rng.randn(b, c, h, w))
    y = nn.Variable.from_numpy_array(rng.randint(0, n_cls, b).reshape(b, 1))
    p = SmallResNet(x, shared=shared)
    loss = F.mean(F.softmax_cross_entropy(p, y))

    # Zerograd, Forward, Backward on the forward graph
    inputs = nn.get_parameters().values()
    [inp.grad.zero() for inp in inputs]
    grad = nn.NdArray.from_numpy_array(
        np.asarray(rng.randn())) if flag_grad_outputs else 1
    if not auto_forward:
        loss.forward()
    loss.backward(grad)

    # Grad
    grad_outputs = grad if flag_grad_outputs else None
    grads = nn.grad([loss], inputs, [grad_outputs])
    if not auto_forward:
        F.sink(*grads, one_input_grad=1).forward()

    # Check between results of var.bacwkard and nn.grad
    backend = ctx.backend[0].split(":")[0]
    if backend == 'cuda':
        pytest.skip('CUDA Convolution N-D is only supported in CUDNN extension')
    for inp, grad in zip(inputs, grads):
        assert_allclose(
            inp.g, grad.d, atol=1e-6)


@pytest.mark.parametrize("seed", [311])
@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("auto_forward", [True, False])
def test_multiple_objectives(seed, ctx, auto_forward):

    # Settings
    nn.set_default_context(ctx)
    nn.set_auto_forward(auto_forward)
    b, c, h, w = 4, 3, 32, 32
    n_cls = 10
    rng = np.random.RandomState(seed)

    # Objecive0
    x0 = nn.Variable.from_numpy_array(
        rng.randn(b, c, h, w)).apply(need_grad=True)
    y0 = F.sigmoid(x0)
    # Objecive1
    x1 = nn.Variable.from_numpy_array(
        rng.randn(b, c, h, w)).apply(need_grad=True)
    y1 = F.tanh(x1)

    # Zerograd, Forward, Backward on the forward graph
    g0 = nn.NdArray.from_numpy_array(rng.randn(*x0.shape))
    g1 = nn.NdArray.from_numpy_array(rng.randn(*x1.shape))
    z = y0 * nn.Variable(g0.shape).apply(data=g0) + y1 * \
        nn.Variable(g1.shape).apply(data=g1)
    inputs = [x0, x1]
    [inp.grad.zero() for inp in inputs]
    if not auto_forward:
        z.forward()
    z.backward()

    # Grad
    inputs = [x0, x1]
    outputs = [y0, y1]
    grad_outputs = [g0, g1]
    grads = nn.grad(outputs, inputs, grad_outputs)
    if not auto_forward:
        F.sink(*grads, one_input_grad=1).forward()

    # Check between results of var.bacwkard and nn.grad
    for inp, grad in zip(inputs, grads):
        assert_allclose(
            inp.g, grad.d, atol=1e-6)


@pytest.mark.parametrize("seed", [311])
@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("auto_forward", [True, False])
@pytest.mark.parametrize("type_grad_outputs", [int, float, np.ndarray, nn.NdArray])
def test_grad_outputs(seed, ctx, auto_forward, type_grad_outputs):

    # Settings
    nn.set_default_context(ctx)
    nn.set_auto_forward(auto_forward)
    b, c, h, w = 4, 3, 32, 32
    n_cls = 10
    rng = np.random.RandomState(seed)

    x = nn.Variable.from_numpy_array(
        rng.randn(b, c, h, w)).apply(need_grad=True)
    y = F.sigmoid(x)

    # Grad outputs
    if type_grad_outputs == int:
        g = rng.randint(-10, 10)
    elif type_grad_outputs == float:
        g = rng.randn()
    elif type_grad_outputs == np.ndarray:
        g = rng.randn(*y.shape)
    elif type_grad_outputs == nn.NdArray:
        g = nn.NdArray.from_numpy_array(rng.randn(*y.shape))

    # Zerograd, Forward, Backward on the forward graph
    inputs = [x]
    [inp.grad.zero() for inp in inputs]
    if not auto_forward:
        y.forward()
    y.backward(g)

    # Grad
    inputs = [x]
    outputs = [y]
    grad_outputs = [g]
    grads = nn.grad(outputs, inputs, grad_outputs)
    if not auto_forward:
        F.sink(*grads, one_input_grad=1).forward()

    # Check between results of var.bacwkard and nn.grad
    for inp, grad in zip(inputs, grads):
        assert_allclose(
            inp.g, grad.d, atol=1e-6)


@pytest.mark.parametrize("seed", [311])
@pytest.mark.parametrize("ctx", ctx_list)
@pytest.mark.parametrize("auto_forward", [True, False])
def test_shared_leaf_variable_basic_arithmetics(seed, ctx, auto_forward):
    def add(x, derivate=0):
        if derivate == 0:
            return x + x + x
        if derivate == 1:
            return 3 * np.ones_like(x)
        if derivate == 2:
            return np.zeros_like(x)

    def sub(x, derivate=0):
        if derivate == 0:
            return x - x - x
        if derivate == 1:
            return -1 * np.ones_like(x)
        if derivate == 2:
            return np.zeros_like(x)

    def mul(x, derivate=0):
        if derivate == 0:
            return x * x * x
        if derivate == 1:
            return 3 * x ** 2
        if derivate == 2:
            return 6 * x

    def div(x, derivate=0):
        if derivate == 0:
            return x / x / x
        if derivate == 1:
            return - x ** -2
        if derivate == 2:
            return 2 * x ** -3

    # Settings
    nn.set_default_context(ctx)
    nn.set_auto_forward(auto_forward)

    for math_type in [add, sub, mul, div]:
        xd = np.random.randn(2, 3) + 0.5
        x = nn.Variable.from_numpy_array(xd).apply(need_grad=True)
        x.grad.zero()
        y = math_type(x)
        # First-order gradient
        dy_dx = nn.grad([y], [x])
        if not auto_forward:
            dy_dx[0].forward()
        assert_allclose(dy_dx[0].d, math_type(xd, 1))
        # Second-order gradient
        dy_dx[0].backward()
        assert_allclose(x.g, math_type(xd, 2))
