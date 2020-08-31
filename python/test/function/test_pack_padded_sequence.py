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

ctxs = list_context('PackPaddedSequence')


def compute_batch_sizes(lengths):
    tmp_lengths = np.copy(lengths)
    batch_sizes = []
    while True:
        c = np.count_nonzero(tmp_lengths > 0)
        if c == 0:
            break
        batch_sizes.append(c)
        tmp_lengths = np.array([l - 1 for l in tmp_lengths])
    return np.array(batch_sizes)


def ref_pack_padded_sequence(padded_sequence, lengths, batch_first):
    batch_sizes = compute_batch_sizes(lengths)
    packed_sequence = []
    for t, batch_size in enumerate(batch_sizes):
        if batch_first:
            packed_sequence_t = padded_sequence[:batch_size, t]
        else:
            packed_sequence_t = padded_sequence[t, :batch_size]
        packed_sequence.append(packed_sequence_t)
    packed_sequence = np.concatenate(packed_sequence, 0)
    return packed_sequence, batch_sizes


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shapes", [
    [(4, ), (3, ), (3, ), (2, ), (2, )],                       # [(T_i, )]
    [(4, 4), (3, 4), (3, 4), (2, 4), (2, 4)],                  # [(T_i, D_1)]
    [(4, 4, 3), (3, 4, 3), (3, 4, 3), (2, 4, 3), (2, 4, 3)],   # [(T_i, D_1, D_2)]
])
@pytest.mark.parametrize("batch_first", [False, True])
def test_pack_padded_sequence_forward_backward(batch_first, shapes, seed, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    sequences = [rng.randn(*shape).astype(np.float32) for shape in shapes]
    padded_sequence = pad_sequence(sequences, batch_first)
    lengths = np.array([seq.shape[0] for seq in sequences])
    inputs = [padded_sequence, lengths]
    func_args = [batch_first]

    function_tester(rng, F.pack_padded_sequence, ref_pack_padded_sequence, inputs,
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
def test_pack_padded_long_sequence_forward_backward(total_length, padding_value,
                                                    batch_first, shapes, seed, ctx, func_name):
    if not func_name.endswith("Cuda"):
        pytest.skip(
            "PackPaddedSequence tests except for Cuda for very long sequence skips.")

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)

    sequences = [rng.randn(*shape).astype(np.float32) for shape in shapes]
    padded_sequence = pad_sequence(sequences, batch_first)
    lengths = np.array([seq.shape[0] for seq in sequences])
    inputs = [padded_sequence, lengths]
    func_args0 = [batch_first]
    func_args1 = [batch_first, padding_value, total_length]

    # Forward
    function_tester(rng, F.pack_padded_sequence, ref_pack_padded_sequence, inputs,
                    ctx=ctx, func_name=func_name, func_args=func_args0,
                    backward=[False, False],
                    atol_f=1e-3, atol_b=1e-2)

    # Backward
    import nnabla as nn
    padded_sequence0 = nn.Variable.from_numpy_array(
        inputs[0]).apply(need_grad=True)
    lengths = nn.Variable.from_numpy_array(inputs[1])
    with nn.context_scope(ctx), nn.auto_forward():
        # Pack backward
        padded_sequence0.g = rng.randn(*padded_sequence0.shape)
        packed_sequence0, batch_sizes = F.pack_padded_sequence(
            padded_sequence0, lengths, *func_args0)
        g = rng.randn(*packed_sequence0.shape)
        packed_sequence0.g = g
        packed_sequence0.parent.backward([padded_sequence0, lengths], [packed_sequence0, batch_sizes],
                                         [False, False])
        # Unpack
        packed_sequence1 = nn.Variable.from_numpy_array(g)
        padded_sequence1, lengths = F.pad_packed_sequence(
            packed_sequence1, batch_sizes, *func_args1)
        # Compare w/o accum
        np.testing.assert_allclose(padded_sequence0.g.flatten(),
                                   padded_sequence1.d.flatten(
                                   )[:np.prod(padded_sequence0.shape)],
                                   atol=1e-4,
                                   err_msg="{} test (w/o accum) with long sequence failed.".format(func_name))
        # Compare w/ accum
        packed_sequence0.parent.backward([padded_sequence0, lengths], [packed_sequence0, batch_sizes],
                                         [True, False])
        np.testing.assert_allclose(padded_sequence0.g.flatten() / 2,
                                   padded_sequence1.d.flatten(
                                   )[:np.prod(padded_sequence0.shape)],
                                   atol=1e-4,
                                   err_msg="{} test (w/ accum) with long sequence failed.".format(func_name))
