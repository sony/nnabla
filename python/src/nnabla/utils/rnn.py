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

import nnabla as nn
import nnabla.functions as F
import numpy as np

from collections import defaultdict


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
