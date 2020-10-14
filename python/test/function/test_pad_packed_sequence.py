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
import nnabla.functions as F
from nbla_test_utils import list_context
from refs import pad_sequence

ctxs = list_context('PadPackedSequence')


def compute_lengths(batch_sizes):
    tmp_batch_sizes = np.copy(batch_sizes)
    lengths = []
    while True:
        c = np.count_nonzero(tmp_batch_sizes > 0)
        if c == 0:
            break
        lengths.append(c)
        tmp_batch_sizes = np.array([b - 1 for b in tmp_batch_sizes])
    return np.array(lengths)


def ref_pad_packed_sequence(packed_sequence, batch_sizes, batch_first, padding, total_length):
    lengths = compute_lengths(batch_sizes)
    B = np.max(batch_sizes)
    T = len(batch_sizes)
    T0 = max(T, total_length)
    D0 = B if batch_first else T0
    D1 = T0 if batch_first else B
    Ds = packed_sequence.shape[1:] if len(packed_sequence.shape) > 1 else ()
    padded_sequence = np.zeros((D0, D1) + Ds)
    b_prev = 0
    b_curr = 0
    for t, b in enumerate(batch_sizes):
        b_curr += b
        packed_sequence_b = packed_sequence[b_prev:b_curr]
        if batch_first:
            padded_sequence[:b, t] = packed_sequence_b
        else:
            padded_sequence[t, :b] = packed_sequence_b
        b_prev = b_curr
    return padded_sequence, lengths


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("Ds", [(), (4, ), (4, 3)])
@pytest.mark.parametrize("batch_sizes", [[5, 3, 1, 1]])
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("padding_value", [0.0])
@pytest.mark.parametrize("total_length", [3, 5])
def test_pad_packed_sequence_forward_backward(total_length, padding_value, batch_first,
                                              batch_sizes, Ds,
                                              seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    packed_sequence = rng.randn(
        *((sum(batch_sizes), ) + Ds)).astype(np.float32)
    batch_sizes = np.array(batch_sizes)
    inputs = [packed_sequence, batch_sizes]
    func_args = [batch_first, padding_value, total_length]
    function_tester(rng, F.pad_packed_sequence, ref_pad_packed_sequence, inputs,
                    ctx=ctx, func_name=func_name, func_args=func_args,
                    backward=[True, False], disable_half_test=False,
                    atol_f=1e-3, atol_b=1e-2)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shapes", [
    [(10000, 4), (5000, 4), (3000, 4), (3000, 4)],        # [(T_i, D_1)]
])
@pytest.mark.parametrize("batch_first", [False])
@pytest.mark.parametrize("padding_value", [0.0])
@pytest.mark.parametrize("total_length", [5000, 10001])
def test_pad_packed_long_sequence_forward_backward(total_length, padding_value,
                                                   batch_first, shapes, seed, ctx, func_name):
    if not func_name.endswith("Cuda"):
        pytest.skip(
            "PadPackedSequence tests except for Cuda for very long sequence skips.")

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    sequences = [rng.randn(*shape).astype(np.float32) for shape in shapes]
    padded_sequence = pad_sequence(sequences, batch_first)
    lengths = np.array([seq.shape[0] for seq in sequences])
    inputs = [padded_sequence, lengths]
    func_args0 = [batch_first]
    func_args1 = [batch_first, padding_value, total_length]

    import nnabla as nn
    padded_sequence0 = nn.Variable.from_numpy_array(
        inputs[0]).apply(need_grad=True)
    lengths = nn.Variable.from_numpy_array(inputs[1])
    with nn.context_scope(ctx), nn.auto_forward():
        # Pack
        packed_sequence0, batch_sizes = F.pack_padded_sequence(
            padded_sequence0, lengths, *func_args0)
    # Forward
    inputs = [packed_sequence0.d, batch_sizes.d]
    function_tester(rng, F.pad_packed_sequence, ref_pad_packed_sequence, inputs,
                    ctx=ctx, func_name=func_name, func_args=func_args1,
                    backward=[False, False],
                    atol_f=1e-3, atol_b=1e-2)

    # Backward
    import nnabla as nn
    packed_sequence0 = packed_sequence0.get_unlinked_variable(need_grad=True)
    with nn.context_scope(ctx), nn.auto_forward():
        # Unpack backward
        packed_sequence0.g = rng.randn(*packed_sequence0.shape)
        padded_sequence0, lengths = F.pad_packed_sequence(
            packed_sequence0, batch_sizes, *func_args1)
        g = rng.randn(*padded_sequence0.shape)
        padded_sequence0.g = g
        padded_sequence0.parent.backward([packed_sequence0, batch_sizes], [padded_sequence0, lengths],
                                         [False, False])
        # Pack
        T = lengths.d[0]  # max_length < total_length
        padded_sequence1 = nn.Variable.from_numpy_array(g[:T, ...])
        packed_sequence1, batch_sizes = F.pack_padded_sequence(
            padded_sequence1, lengths, *func_args0)
        # Compare w/o accum
        np.testing.assert_allclose(packed_sequence0.g.flatten(),
                                   packed_sequence1.d.flatten(),
                                   atol=1e-4,
                                   err_msg="{} test (w/o accum) with long sequence failed.".format(func_name))
        padded_sequence0.parent.backward([packed_sequence0, batch_sizes], [padded_sequence0, lengths],
                                         [True, False])
        np.testing.assert_allclose(packed_sequence0.g.flatten() / 2,
                                   packed_sequence1.d.flatten(),
                                   atol=1e-4,
                                   err_msg="{} test (w/ accum) with long sequence failed.".format(func_name))
