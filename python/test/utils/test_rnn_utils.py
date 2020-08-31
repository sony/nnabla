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

import nnabla as nn
import nnabla.functions as F
import numpy as np
import nnabla.utils.rnn as rnn_utils

from nbla_test_utils import list_context

# Proxy to use the context
ctxs = list_context('PackPaddedSequence')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("shapes", [
    [(4, ), (3, ), (3, ), (2, ), (2, )],                       # [(T_i, )]
    [(4, 4), (3, 4), (3, 4), (2, 4), (2, 4)],                  # [(T_i, D_1)]
    [(4, 4, 3), (3, 4, 3), (3, 4, 3), (2, 4, 3), (2, 4, 3)],   # [(T_i, D_1, D_2)]
])
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("enforce_sorted", [False, True])
@pytest.mark.parametrize("total_length", [None, 5])
def test_pack_and_unpack(total_length, enforce_sorted, batch_first, shapes, seed,
                         ctx, func_name):
    rng = np.random.RandomState(seed)

    sequences = [rng.randn(*shape).astype(np.float32) for shape in shapes]
    if not enforce_sorted:
        indices = rng.permutation(len(sequences))
        sequences = [sequences[i] for i in indices]
    sequences = [nn.Variable.from_numpy_array(s) for s in sequences]

    with nn.context_scope(ctx), nn.auto_forward():
        padded_sequence0 = rnn_utils.pad_sequence(sequences, batch_first)
        packed_sequence = rnn_utils.pack_sequence(sequences, batch_first,
                                                  enforce_sorted)
        padded_sequence, _ = rnn_utils.pad_packed_sequence(packed_sequence,
                                                           batch_first,
                                                           total_length)
        if total_length is not None:
            batch_sizes = packed_sequence.batch_sizes
            T = batch_sizes.shape[0]
            padded_sequence = padded_sequence[:, :T, ...] if batch_first else \
                padded_sequence[:T, :, ...]

        np.testing.assert_allclose(padded_sequence0.d,
                                   padded_sequence.d)
