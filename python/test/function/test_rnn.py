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
from nbla_test_utils import list_context

ctxs = list_context('RNN')


def rnn(x, h, w, b, nonlinearity, with_bias):
    hidden_size = h.shape[1]
    xh = F.concatenate(*(x, h), axis=1)
    b_ = None
    if with_bias:
        b_ = b
    h_t = F.affine(xh, F.transpose(w, (1, 0)), b_)
    if nonlinearity == 'tanh':
        h_t = F.tanh(h_t)
    elif nonlinearity == 'relu':
        h_t = F.relu(h_t)

    return h_t


def create_fixed_length_rnn(xs0, h0, w0, w, b, num_layers, nonlinearity, num_directions, with_bias):
    # xs : [T, B, I]
    # h0 : [L, D, B, H]
    # c0 : [L, D, B, H]
    # w0 : [D, H, I+H]
    # w : [L-1, D, H, D * H + H]
    # b : [L, D, H]

    batch_size = xs0.shape[1]
    hidden_size = h0.shape[3]

    if xs0.shape[0] == 1:
        xs = [xs0[0]]
    else:
        xs = F.split(xs0, axis=0)
    hn = []
    for i in range(num_layers):
        wi = w0
        if i > 0:
            wi = w[i - 1]
        # wi : [D, H, ?]
        # Forward direction
        hif = h0[i, 0]  # [B, H]
        wif = wi[0]
        bif = None
        if with_bias:
            bif = b[i, 0]
        hs = []
        for j, x in enumerate(xs):
            # x : [B, I]
            hif = rnn(x, hif, wif, bif, nonlinearity, with_bias)
            hs.append(hif)
        hn.append(hif)

        if num_directions == 1:
            xs = hs
            continue

        # Backward direction
        hib = h0[i, 1]  # [B, H]
        wib = wi[1]
        bib = None
        if with_bias:
            bib = b[i, 1]
        for k, x, in enumerate(reversed(xs)):
            j = len(xs) - 1 - k
            # x : [B, I]
            hib = rnn(x, hib, wib, bib, nonlinearity, with_bias)
            hs[j] = F.concatenate(hs[j], hib, axis=1)
        hn.append(hib)
        xs = hs

    ys = xs  # list of [B, HD]
    ys = F.stack(*ys, axis=0)  # [T, B, HD]
    hn = F.reshape(F.stack(*hn, axis=0), (num_layers, num_directions,
                                          batch_size, hidden_size))  # LD list of [B, H] --> [L, D, B, H]
    return ys, hn


def execute_fixed_length_rnn(xs_np, h0_np, w0_np, w_np, b_np, num_layers=1, nonlinearity='tanh', dropout=0.0, bidirectional=False, training=True):
    # Inputs are numpy arrays
    num_directions = 2 if bidirectional else 1
    seq_len = xs_np.shape[0]
    batch_size = xs_np.shape[1]
    hidden_size = h0_np.shape[3]

    xs = nn.Variable.from_numpy_array(xs_np)
    h0 = nn.Variable.from_numpy_array(h0_np)
    w0 = nn.Variable.from_numpy_array(w0_np)
    w = None
    b = None
    with_bias = False
    if num_layers > 1:
        w = nn.Variable.from_numpy_array(w_np)
    if type(b_np) is np.ndarray:
        b = nn.Variable.from_numpy_array(b_np)
        with_bias = True

    ys, hn = create_fixed_length_rnn(
        xs, h0, w0, w, b, num_layers, nonlinearity, num_directions, with_bias)  # returns Variables

    dummy = F.sink(ys, hn)
    dummy.forward()

    # returns numpy arrays
    ys = F.reshape(ys, (seq_len, batch_size, num_directions * hidden_size))
    ys.forward()
    return ys.d, hn.d


def get_rnn_grad(xs_np, h0_np, w0_np, w_np, b_np, dy, dh, num_layers=1, nonlinearity='tanh', dropout=0.0, bidirectional=False, training=True):
    # Inputs are numpy arrays
    num_directions = 2 if bidirectional else 1
    seq_len = xs_np.shape[0]
    batch_size = xs_np.shape[1]
    hidden_size = h0_np.shape[3]

    xs = nn.Variable.from_numpy_array(xs_np, need_grad=True)
    h0 = nn.Variable.from_numpy_array(h0_np, need_grad=True)
    w0 = nn.Variable.from_numpy_array(w0_np, need_grad=True)
    w = None
    b = None
    with_bias = False
    if num_layers > 1:
        w = nn.Variable.from_numpy_array(w_np, need_grad=True)
    if type(b_np) == np.ndarray:
        b = nn.Variable.from_numpy_array(b_np, need_grad=True)
        with_bias = True
    xs.grad.zero()
    h0.grad.zero()
    w0.grad.zero()
    if num_layers > 1:
        w.grad.zero()
    if with_bias:
        b.grad.zero()

    ys, hn = create_fixed_length_rnn(
        xs, h0, w0, w, b, num_layers, nonlinearity, num_directions, with_bias)  # returns Variables

    dummy = F.sink(ys, hn, one_input_grad=False)
    dummy.forward()
    ys.g = np.reshape(dy, ys.shape)
    hn.g = dh
    dummy.backward()

    if num_layers > 1 and with_bias:
        return np.concatenate((xs.g.flat, h0.g.flat, w0.g.flat, w.g.flat, b.g.flat))
    elif num_layers > 1 and not with_bias:
        return np.concatenate((xs.g.flat, h0.g.flat, w0.g.flat, w.g.flat))
    elif num_layers == 1 and with_bias:
        return np.concatenate((xs.g.flat, h0.g.flat, w0.g.flat, b.g.flat))
    else:
        return np.concatenate((xs.g.flat, h0.g.flat, w0.g.flat))


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [100])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("seq_len", [2])
@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("input_size", [2])
@pytest.mark.parametrize("hidden_size", [3])
@pytest.mark.parametrize("with_bias", [True, False])
def test_rnn(seed, num_layers, nonlinearity, dropout, bidirectional, training, seq_len, batch_size, input_size, hidden_size, with_bias, ctx, func_name):
    from nbla_test_utils import function_tester

    if func_name == "RNN":
        pytest.skip("Not implemented in CPU.")

    rng = np.random.RandomState(seed)
    num_directions = 1
    if bidirectional:
        num_directions = 2
    inputs = [rng.randn(seq_len, batch_size, input_size).astype(np.float32)]
    inputs += [rng.randn(num_layers, num_directions,
                         batch_size, hidden_size).astype(np.float32)]
    inputs += [rng.randn(num_directions, hidden_size,
                         input_size + hidden_size)]
    if num_layers > 1:
        inputs += [rng.randn(max(1, num_layers-1), num_directions, hidden_size,
                             num_directions*hidden_size + hidden_size).astype(np.float32)]
    else:
        inputs += [None]
    if with_bias:
        inputs += [rng.randn(num_layers, num_directions,
                             hidden_size).astype(np.float32)]
    else:
        inputs += [None]

    backward = [False for _ in inputs]
    if training:
        backward = [True for _ in inputs]

    function_tester(rng, F.rnn, execute_fixed_length_rnn, inputs, func_kwargs=dict(
                    num_layers=num_layers, nonlinearity=nonlinearity, dropout=dropout, bidirectional=bidirectional, training=training), atol_f=1e-4, atol_b=1e-3, dstep=1e-3, backward=backward, ctx=ctx, func_name=func_name, ref_grad=get_rnn_grad, disable_half_test=True)


@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("bidirectional", [False])
@pytest.mark.parametrize("seq_len", [2])
@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("input_size", [2])
@pytest.mark.parametrize("hidden_size", [3])
@pytest.mark.parametrize("ctx, func_name", ctxs)
def test_inference_backward(num_layers, bidirectional, seq_len, batch_size, input_size, hidden_size, ctx, func_name):
    with nn.context_scope(ctx):
        if func_name == "RNN":
            pytest.skip("Not implemented in CPU.")
        num_directions = 1
        if bidirectional:
            num_directions = 2

        x = nn.Variable((seq_len, batch_size, input_size), need_grad=True)
        h = nn.Variable((num_layers, num_directions,
                         batch_size, hidden_size), need_grad=True)
        w0 = nn.Variable((num_directions, hidden_size,
                          input_size + hidden_size), need_grad=True)
        w = nn.Variable((max(1, num_layers-1), num_directions, hidden_size,
                         num_directions*hidden_size + hidden_size), need_grad=True)
        b = nn.Variable((num_layers, num_directions,
                         hidden_size), need_grad=True)
        y, hn = F.rnn(x, h, w0, w, b, num_layers=num_layers,
                      nonlinearity="tanh", training=False)
        y.forward()
    with pytest.raises(RuntimeError) as e_info:
        y.backward()
