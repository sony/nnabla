# Copyright 2020,2021 Sony Corporation.
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

import nnabla as nn
import nnabla.functions as F
import numpy as np


class PackedSequence(object):
    """
    Args:
      data (:obj:`nnabla.Variable`): Packed sequence.
      batch_sizes (:obj:`nnabla.Variable`): Batch size for each time step and always resides in CPU.
      sorted_indices (:obj:`nnabla.Variable`): Sorted indices to reconstruct the original sequences.
      unsorted_indices (:obj:`nnabla.Variable`): Unsorted indices to reconstruct the original sequences.
    """

    def __init__(self):
        self.data = None
        self.batch_sizes = None
        self.sorted_indices = None
        self.unsorted_indices = None

    def __str__(self):
        return "data={}, batch_sizes={}, "\
          "sorted_indices={}, unsorted_indices={self.unsorted_indices}"\
          .format(self.data, self.batch_sizes, self.sorted_indices, self.unsorted_indices)


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    """Pad a list of variable-length Variables.

    This method stacks a list of variable-length :obj:`nnabla.Variable` s with the padding_value.

    :math:`T_i` is the length of the :math:`i`-th Variable in the sequences.
    :math:`B` is the batch size equal to the length of the sequences. 
    :math:`T` is the max of :math:`T_i` for all :math:`i`. 
    :math:`*` is the remaining dimensions including none.

    .. note::
      This function **must** be used the dynamic computation mode.

    Example:

    .. code-block:: python

      import numpy as np
      import nnabla as nn
      import nnabla.functions as F
      import nnabla.utils.rnn as rnn_utils

      nn.set_auto_forward(True)

      l2v = lambda ldata: nn.Variable.from_numpy_array(np.asarray(ldata))
      a = l2v([1, 1, 1, 1])
      b = l2v([2, 2, 2])
      c = l2v([2, 2, 2])
      d = l2v([3, 3])
      e = l2v([3, 3])
      sequences = [a, b, c, d, e]

      padded_sequence = rnn_utils.pad_sequence(sequences)
      print(padded_sequence.d)

    Args:
      sequences (list of :obj:`nnabla.Variable`): Sequence of the variable of (:math:`T_i`, :math:`*`) shape. 
      batch_first (bool): If False, output is of (:math:`T`, :math:`B`, :math:`*`) shape, 
                          otherwise (:math:`B`, :math:`T`, :math:`*`).
      padding_value (float): Padding value.

    Returns: 
      :obj:`nnabla.Variable` of (:math:`T`, :math:`B`, :math:`*`) or (:math:`B`, :math:`T`, :math:`*`) shape
    """

    B = len(sequences)
    T = max([e.shape[0] for e in sequences])
    shape0 = (B, T) if batch_first else (T, B)
    shape1 = sequences[0].shape[1:]

    padded_sequence = F.constant(padding_value, shape0 + shape1)
    for b, s in enumerate(sequences):
        l = s.shape[0]
        if batch_first:
            padded_sequence[b, :l, ...] = s
        else:
            padded_sequence[:l, b, ...] = s
    return padded_sequence


def pack_padded_sequence(padded_sequence, lengths, batch_first=False, enforce_sorted=True):
    r"""Pack a padded variable-length sequences.

    This method packs a padded variable-length sequences.

    :math:`T` is the max length over the lengths of sequences.
    :math:`B` is the batch size equal to the length of the sequences.     
    :math:`*` is the remaining dimensions including none.

    .. note::
      This function **must** be used the dynamic computation mode.


    Example:

    .. code-block:: python

      import numpy as np
      import nnabla as nn
      import nnabla.functions as F
      import nnabla.utils.rnn as rnn_utils

      nn.set_auto_forward(True)

      l2v = lambda ldata: nn.Variable.from_numpy_array(np.asarray(ldata))
      a = l2v([1, 1, 1, 1])
      b = l2v([2, 2, 2])
      c = l2v([2, 2, 2])
      d = l2v([3, 3])
      e = l2v([3, 3])
      sequences = [a, b, c, d, e]
      lengths = l2v([seq.shape[0] for seq in sequences])

      padded_sequence = rnn_utils.pad_sequence(sequences)
      print(padded_sequence.d)

      packed_sequence = rnn_utils.pack_padded_sequence(padded_sequence, lengths)
      print(packed_sequence.data.d)
      print(packed_sequence.batch_sizes.d)

    Args: 
      padded_sequence (:obj:`nnabla.Variable`): Padded sequence of (:math:`T \times B \times *`)
                                                or (:math:`B \times T \times *`) shape.
      lengths (:obj:`nnabla.Variable`): Sequence length for each batch and always resides in CPU.
      batch_first (bool): `padded_sequence` is of (:math:`T`, :math:`B`, :math:`*`) shape if False,
                          otherwise (:math:`B`, :math:`T`, :math:`*`).
      enforce_sorted (bool): Sequences are sorted by the length in a decreasing order if True. Default is True.

    Returns: 
        :obj:`PackedSequence`
    """
    if enforce_sorted:
        sorted_indices = None
        unsorted_indices = None
    else:
        # TODO: replace cuda context when the bug fix of the sort
        with nn.context_scope(nn.Context()):
            lengths, sorted_indices = F.sort(
                lengths, axis=0, reverse=True, with_index=True)

        B = sorted_indices.shape[0]
        unsorted_indices = F.scatter_nd(
            F.arange(0, B), sorted_indices.reshape((1, B)), shape=(B, ))
        axis = 0 if batch_first else 1
        padded_sequence = F.gather(padded_sequence, sorted_indices, axis)

    packed_sequence, batch_sizes = F.pack_padded_sequence(
        padded_sequence, lengths, batch_first)
    packed_sequence0 = PackedSequence()
    packed_sequence0.data = packed_sequence
    packed_sequence0.batch_sizes = batch_sizes
    packed_sequence0.sorted_indices = sorted_indices
    packed_sequence0.unsorted_indices = unsorted_indices

    return packed_sequence0


def pack_sequence(sequences, batch_first=False, enforce_sorted=True):
    """Pack a list of variable-length Variables.

    This method packs a list of variable-length Variables.

    :math:`T_i` is the length of the :math:`i`-th Variable in the sequences. 
    :math:`T` is the max of :math:`T_i` for all :math:`i`.
    :math:`B` is the batch size equal to the length of the sequences.     
    :math:`*` is the remaining dimensions including none.

    .. note::
      This function **must** be used the dynamic computation mode.

    Example:

    .. code-block:: python

      import numpy as np
      import nnabla as nn
      import nnabla.functions as F
      import nnabla.utils.rnn as rnn_utils

      nn.set_auto_forward(True)

      l2v = lambda ldata: nn.Variable.from_numpy_array(np.asarray(ldata))
      a = l2v([3, 3])
      b = l2v([2, 2, 2])
      c = l2v([2, 2, 2])
      d = l2v([1, 1, 1, 1])
      e = l2v([3, 3])
      sequences = [a, b, c, d, e]

      packed_sequence = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
      print(packed_sequence.data.d)
      print(packed_sequence.batch_sizes.d)

    Args: 
      sequences (list of :obj:`nnabla.Variable`): List of :obj:`nnabla.Variable` of (:math:`T_i`, :math:`*`) shape. 
      enforce_sorted (bool): Sequences are sorted by the length in a decreasing order if True. Default is True.

    Returns: 
        :obj:`PackedSequence`: packed_sequence
    """
    pad_sequences = pad_sequence(sequences)
    lengths = np.array([sequence.shape[0] for sequence in sequences])
    lengths = nn.Variable.from_numpy_array(lengths)
    packed_sequence = pack_padded_sequence(
        pad_sequences, lengths, enforce_sorted=enforce_sorted)
    return packed_sequence


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    """Pad packed sequence.

    This method unpacks the packed sequqnce and pad it, the inverse operation of :func:`pack_padded_sequence`.

    :math:`T_i` is the length of the :math:`i`-th Variable in the sequences.
    :math:`B` is the batch size equal to the length of the sequences. 
    :math:`T` is the max of :math:`T_i` for all :math:`i`. 
    :math:`*` is the remaining dimensions including none.

    .. note::
      This function **must** be used the dynamic computation mode.

    Example:

    .. code-block:: python

      import numpy as np
      import nnabla as nn
      import nnabla.functions as F
      import nnabla.utils.rnn as rnn_utils

      nn.set_auto_forward(True)

      l2v = lambda ldata: nn.Variable.from_numpy_array(np.asarray(ldata))
      a = l2v([3, 3])
      b = l2v([2, 2, 2])
      c = l2v([2, 2, 2])
      d = l2v([1, 1, 1, 1])
      e = l2v([3, 3])
      sequences = [a, b, c, d, e]

      packed_sequence = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
      print(packed_sequence.data.d)
      print(packed_sequence.batch_sizes.d)

      padded_sequence, lengths = rnn_utils.pad_packed_sequence(packed_sequence)
      print(padded_sequence.d)
      print(lengths.d)

    Args: 
      sequence (:obj:`PackedSequence`): PackedSequence.
      batch_first (bool): If False, output is of (:math:`T`, :math:`B`, :math:`*`) shape,
                          otherwise (:math:`B`, :math:`T`, :math:`*`).
      padding_value (float): Padding value.
      total_length (int): If not None, the outputs are padded up to the `total_length`.
                          If the `total_length` is less than the max length in the `sequences`,
                          the error is thrown.
                          This is normally used in the distributed training to align with 
                          the longest sequence in a distributed system.

    Returns:
      :obj:`nnabla.Variable` of (:math:`T`, :math:`B`, :math:`*`) or (:math:`B`, :math:`T`, :math:`*`) shape
    """
    packed_sequence = sequence.data
    batch_sizes = sequence.batch_sizes
    sorted_indices = sequence.sorted_indices
    unsorted_indices = sequence.unsorted_indices

    T = batch_sizes.shape[0]
    if total_length is not None:
        if total_length < T:
            raise ValueError("`total length ({})` must be greater than or equal to the maximum length ({})."
                             .format(total_length, T))

    padded_sequence, lengths = F.pad_packed_sequence(packed_sequence, batch_sizes,
                                                     batch_first, padding_value,
                                                     total_length)
    if unsorted_indices is not None:
        axis = 0 if batch_first else 1
        padded_sequence = F.gather(padded_sequence, unsorted_indices, axis)
        lengths = lengths[unsorted_indices]
    return padded_sequence, lengths


def _rnn(x, h, w, b, nonlinearity, with_bias):
    """RNN cell.
    Args:
        x (:obj:`~nnabla.Variable`): Input data.
        h (:obj:`~nnabla.Variable`): Hidden state.
        w (:obj:`~nnabla.Variable`): Weight.
        b (:obj:`~nnabla.Variable`): Bias.
        nonlinearity (str): "tanh" or "relu".
        with_bias (bool): Include the bias or not.
    """
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


def _create_fixed_length_rnn(xs0, h0, w0, w, b, num_layers, nonlinearity, num_directions, with_bias):
    """NStepRNNCells over time and over layers.
    Args:
        xs0 (:obj:`~nnabla.Variable`): Input data with [T, B, I]  shape.
        h0 (:obj:`~nnabla.Variable`): Hidden states with [L, D, B, H] shape.
        w0 (:obj:`~nnabla.Variable`): Weights at the first layer with [D, H, I+H] shape.
        w (:obj:`~nnabla.Variable`): Weights with [L-1, D, H, D * H + H]  shape at layers other than the first layer.
        b (:obj:`~nnabla.Variable`): Biases with [L, D, H] shape.
        num_layers (int): Number of layers.
        nonlinearity (str): "tanh" or "relu".
        num_directions (int): "tanh" or "relu".
        with_bias (bool): Include the bias or not.
    """
    # xs : [T, B, I]
    # h0 : [L, D, B, H]
    # w0 : [D, H, I+H]
    # w : [L-1, D, H, D * H + H]
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
            hif = _rnn(x, hif, wif, bif, nonlinearity, with_bias)
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
            hib = _rnn(x, hib, wib, bib, nonlinearity, with_bias)
            hs[j] = F.concatenate(hs[j], hib, axis=1)
        hn.append(hib)
        xs = hs

    ys = xs  # list of [B, HD]
    ys = F.stack(*ys, axis=0)  # [T, B, HD]
    hn = F.reshape(F.stack(*hn, axis=0), (num_layers, num_directions,
                                          batch_size, hidden_size))  # LD list of [B, H] --> [L, D, B, H]
    return ys, hn


def _gru(x, h, w, b, with_bias):
    """GRU cell.
    Args:
        x (:obj:`~nnabla.Variable`): Input data.
        h (:obj:`~nnabla.Variable`): Hidden state.
        w (:obj:`~nnabla.Variable`): Weight.
        b (:obj:`~nnabla.Variable`): Bias.
        with_bias (bool): Include the bias or not.
    """
    hidden_size = h.shape[1]
    xh = F.concatenate(*(x, h), axis=1)
    w0, w1, w2 = F.split(w, axis=0)
    b0 = b1 = b2 = b3 = None
    if with_bias:
        b0, b1, b2, b3 = F.split(b, axis=0)
    r_t = F.sigmoid(F.affine(xh, F.transpose(w0, (1, 0)), b0))
    z_t = F.sigmoid(F.affine(xh, F.transpose(w1, (1, 0)), b1))

    w2_0 = w2[:, :w2.shape[1]-hidden_size]
    w2_1 = w2[:, w2.shape[1]-hidden_size:]
    n_t = F.tanh(F.affine(x, F.transpose(w2_0, (1, 0)), b2) +
                 r_t*F.affine(h, F.transpose(w2_1, (1, 0)), b3))
    h_t = (1-z_t)*n_t + z_t*h

    return h_t


def _create_fixed_length_gru(xs0, h0, w0, w, b, num_layers, num_directions, with_bias):
    """NStepGRUCells over time and over layers.
    Args:
        xs0 (:obj:`~nnabla.Variable`): Input data with [T, B, I]  shape.
        h0 (:obj:`~nnabla.Variable`): Hidden states with [L, D, B, H] shape.
        w0 (:obj:`~nnabla.Variable`): Weights at the first layer with [D, 3, H, I+H] shape.
        w (:obj:`~nnabla.Variable`): Weights with [L-1, D, 3, H, D * H + H] shape at layers other than the first layer.
        b (:obj:`~nnabla.Variable`): Biases with [L, D, 3, H] shape.
        num_layers (int): Number of layers.
        num_directions (int): "tanh" or "relu".
        with_bias (bool): Include the bias or not.
    """
    # xs : [T, B, I]
    # h0 : [L, D, B, H]
    # w0 : [D, 3, H, I+H]
    # w : [L-1, D, 3, H, D * H + H]
    # b : [L, D, 3, H]

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
        # wi : [D, 3, H, ?]
        # Forward direction
        hif = h0[i, 0]  # [B, H]
        wif = wi[0]
        bif = None
        if with_bias:
            bif = b[i, 0]
        hs = []
        for j, x in enumerate(xs):
            # x : [B, I]
            hif = _gru(x, hif, wif, bif, with_bias)
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
            hib = _gru(x, hib, wib, bib, with_bias)
            hs[j] = F.concatenate(hs[j], hib, axis=1)
        hn.append(hib)
        xs = hs

    ys = xs  # list of [B, HD]
    ys = F.stack(*ys, axis=0)  # [T, B, HD]
    hn = F.reshape(F.stack(*hn, axis=0), (num_layers, num_directions,
                                          batch_size, hidden_size))  # LD list of [B, H] --> [L, D, B, H]
    return ys, hn


def _lstm(x, h, c, w, b, with_bias):
    """LSTM cell.
    Args:
        x (:obj:`~nnabla.Variable`): Input data.
        h (:obj:`~nnabla.Variable`): Short-term state.
        c (:obj:`~nnabla.Variable`): Long-term state.
        w (:obj:`~nnabla.Variable`): Weight.
        b (:obj:`~nnabla.Variable`): Bias.
        with_bias (bool): Include the bias or not.
    """
    hidden_size = h.shape[1]
    xh = F.concatenate(*(x, h), axis=1)
    w0, w1, w2, w3 = F.split(w, axis=0)
    b0 = b1 = b2 = b3 = None
    if with_bias:
        b0, b1, b2, b3 = F.split(b, axis=0)
    i_t = F.affine(xh, F.transpose(w0, (1, 0)), b0)
    f_t = F.affine(xh, F.transpose(w1, (1, 0)), b1)
    g_t = F.affine(xh, F.transpose(w2, (1, 0)), b2)
    o_t = F.affine(xh, F.transpose(w3, (1, 0)), b3)
    c_t = F.sigmoid(f_t) * c + F.sigmoid(i_t) * F.tanh(g_t)
    h_t = F.sigmoid(o_t) * F.tanh(c_t)

    return h_t, c_t


def _create_fixed_length_lstm(xs0, h0, c0, w0, w, b, num_layers, num_directions, with_bias):
    """NStepGRUCells over time and over layers.
    Args:
        xs0 (:obj:`~nnabla.Variable`): Input data with [T, B, I]  shape.
        h0 (:obj:`~nnabla.Variable`): Short-term states with [L, D, B, H] shape.
        c0 (:obj:`~nnabla.Variable`): Long-term states with [L, D, B, H] shape.
        w0 (:obj:`~nnabla.Variable`): Weights at the first layer with [D, 4, H, I+H] shape.
        w (:obj:`~nnabla.Variable`): Weights with [L-1, D, 4, H, D * H + H] shape at layers other than the first layer.
        b (:obj:`~nnabla.Variable`): Biases with [L, D, 4*H] shape.
        num_layers (int): Number of layers.
        num_directions (int): "tanh" or "relu".
        with_bias (bool): Include the bias or not.
    """
    # xs : [T, B, I]
    # h0 : [L, D, B, H]
    # c0 : [L, D, B, H]
    # w0 : [D, 4, H, I+H]
    # w : [L-1, D, 4, H, D * H + H]
    # b : [L, D, 4*H]

    batch_size = xs0.shape[1]
    hidden_size = h0.shape[3]

    if xs0.shape[0] == 1:
        xs = [xs0[0]]
    else:
        xs = F.split(xs0, axis=0)
    hn = []
    cn = []
    for i in range(num_layers):
        wi = w0
        if i > 0:
            wi = w[i - 1]
        # wi : [D, 4, H, ?]
        # Forward direction
        hif = h0[i, 0]  # [B, H]
        cif = c0[i, 0]  # [B, H]
        wif = wi[0]
        bif = None
        if with_bias:
            bif = b[i, 0]
        hs = []
        for j, x in enumerate(xs):
            # x : [B, I]
            hif, cif = _lstm(x, hif, cif, wif, bif, with_bias)
            hs.append(hif)
        hn.append(hif)
        cn.append(cif)

        if num_directions == 1:
            xs = hs
            continue

        # Backward direction
        hib = h0[i, 1]  # [B, H]
        cib = c0[i, 1]  # [B, H]
        wib = wi[1]
        bib = None
        if with_bias:
            bib = b[i, 1]
        for k, x, in enumerate(reversed(xs)):
            j = len(xs) - 1 - k
            # x : [B, I]
            hib, cib = _lstm(x, hib, cib, wib, bib, with_bias)
            hs[j] = F.concatenate(hs[j], hib, axis=1)
        hn.append(hib)
        cn.append(cib)
        xs = hs

    ys = xs  # list of [B, HD]
    ys = F.stack(*ys, axis=0)  # [T, B, HD]
    hn = F.reshape(F.stack(*hn, axis=0), (num_layers, num_directions,
                                          batch_size, hidden_size))  # LD list of [B, H] --> [L, D, B, H]
    cn = F.reshape(F.stack(*cn, axis=0), (num_layers, num_directions,
                                          batch_size, hidden_size))  # LD list of [B, H] --> [L, D, B, H]
    return ys, hn, cn
