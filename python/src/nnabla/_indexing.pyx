from _variable cimport Variable
from _nd_array cimport NdArray

import numpy as np
import itertools


def advanced_indexing(self, key):
    """Advance numpy-like indexing. Returns a :class:`nnabla.Variable` or
    :class:`nnabla.NdArray` depending on the type of `self`.
    """
    import nnabla.functions as F
    idx = [val for val in key if val is not None]
    newaxes = tuple(1 for k in key if k is None)

    def is_numpy_array(a):
        return isinstance(a, np.ndarray)

    if len(idx) == 1 and isinstance(idx[0], (list, tuple)):
        a = np.array(idx[0])
        if a.dtype == np.bool:
            idx[0] = a

    if len(idx) == 1 and is_numpy_array(idx[0]) and idx[0].dtype == np.bool:
        array = F.gather_nd(self, np.vstack(np.nonzero(idx[0])))

    elif all(isinstance(k, (NdArray, Variable)) for k in idx):
        array = F.gather_nd(self, F.stack(*idx))

    elif all(isinstance(k, (int, tuple, list, np.ndarray)) for k in idx):
        array = F.gather_nd(self, np.array(idx))

    else:
        indices = ', '.join(repr(val) for val in key)
        message = "mixed advanced indexing: {}".format(indices)
        raise NotImplementedError(message)

    if len(newaxes) > 0:
        count = itertools.count()
        shape = [1 if k is None else array.shape[next(count)] for k in key]
        array = F.reshape(array, shape)

    return array


def basic_indexing(self, key):
    """Basic numpy-like indexing. Returns a :class:`nnabla.Variable` or
    :class:`nnabla.NdArray` depending on the type of `self`.
    """
    import nnabla.functions as F
    idx = [val for val in key if val is not None]

    if len(idx) == 0:
        idx = self.ndim * [slice(None)]
        key.extend(idx)

    if Ellipsis in idx:
        ins = self.ndim - len(idx) + 1
        pos = idx.index(Ellipsis)
        idx[pos:pos+1] = ins * [slice(None)]
        pos = key.index(Ellipsis)
        key[pos:pos+1] = ins * [slice(None)]

    if Ellipsis in idx:
        raise IndexError("an index can only have a single ellipsis ('...')")

    if len(idx) > self.ndim:
        raise IndexError("too many indices for {0}-D array".format(self.ndim))

    idx.extend((self.ndim - len(idx)) * [slice(None)])
    key.extend((self.ndim - len(key)) * [slice(None)])

    for i, s in enumerate(self.shape):
        if isinstance(idx[i], int):
            if 0 <= idx[i] < s:
                idx[i] = slice(idx[i], idx[i] + 1, 1)
            elif -s <= idx[i] < 0:
                idx[i] = slice(s + idx[i], s + idx[i] + 1, 1)
            else:
                msg = "index {0} is out of bounds for axis {1} with size {2}"
                raise IndexError(msg.format(idx[i], i, s))

    start, stop, step = zip(*((s.start, s.stop, s.step) for s in idx))
    sliced_array = F.slice(self, start, stop, step)
    sliced_shape = list(sliced_array.shape)
    result_shape = []

    for entry in key:
        if entry is None:
            result_shape.append(1)
        elif isinstance(entry, int):
            sliced_shape.pop(0)
        else:
            result_shape.append(sliced_shape.pop(0))

    return F.reshape(sliced_array, result_shape)


cdef object getitem(object self, object key):
    """Return self[key]."""
    assert isinstance(self, (NdArray, Variable))

    if isinstance(key, (list, np.ndarray, NdArray, Variable)):
        return advanced_indexing(self, [key])

    elif isinstance(key, (slice, int, type(Ellipsis), type(None))):
        return basic_indexing(self, [key])

    elif isinstance(key, tuple):
        sequence_types = (tuple, list, np.ndarray, NdArray, Variable)
        if any(isinstance(k, sequence_types) for k in key):
            return advanced_indexing(self, key)
        else:
            return basic_indexing(self, list(key))

    else:
        raise TypeError("invalid index {}".format(repr(key)))
