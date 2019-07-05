from _variable cimport Variable
from _nd_array cimport NdArray

import numpy as np
import itertools
import sys


def is_numpy_array(a):
    return isinstance(a, np.ndarray)


def advanced_getitem(self, key):
    """Advanced numpy-like indexing where the `key` object contains
    integer or boolean arrays. Returns the indexed sub-space of `self`
    as a :class:`nnabla.Variable` or :class:`nnabla.NdArray` depending
    on the type of `self`.
    """
    import nnabla.functions as F
    idx = [val for val in key if val is not np.newaxis]
    newaxes = list(1 if k is None else 0 for k in key)

    if len(idx) == 1 and isinstance(idx[0], (list, tuple)):
        a = np.array(idx[0])
        if a.dtype == np.bool:
            idx[0] = a

    if len(idx) == 1 and is_numpy_array(idx[0]) and idx[0].dtype == np.bool:
        result = F.gather_nd(self, np.vstack(np.nonzero(idx[0])))

    elif all(isinstance(k, (NdArray, Variable)) for k in idx):
        result = F.gather_nd(self, F.stack(*idx))

    elif all(isinstance(k, (int, tuple, list, np.ndarray)) for k in idx):
        result = F.gather_nd(self, np.array(idx))

    else:
        indices = ', '.join(repr(val) for val in key)
        message = "mixed advanced indexing: {}".format(indices)
        raise NotImplementedError(message)

    if sum(newaxes) > 0:
        shape = newaxes[:newaxes.index(0)]  # leading new axes
        shape.append(result.shape[0])       # selected subspace
        shape.extend(sum(newaxes[newaxes.index(0):]) * (1,))  # other new axes
        shape.extend(result.shape[1:])      # subspace dimensions
        result = F.reshape(result, shape)

    return result


def basic_getitem(self, key):
    """Basic numpy-like indexing where the `key` object contains only slices,
    integers, Ellipsis or None (numpy.newaxis). Returns the indexed sub-space
    of `self` as a :class:`nnabla.Variable` or :class:`nnabla.NdArray`
    depending on the type of `self`.
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
    if sys.version_info.major < 3:
        stop = tuple(None if v == sys.maxsize else v for v in stop)
    array = F.slice(self, start, stop, step)

    count = itertools.count()
    shape = []
    for k in key:
        if isinstance(k, int):
            next(count)  # integer axis gets removed in basic indexing
        else:
            shape.append(1 if k is None else array.shape[next(count)])

    if array.shape != tuple(shape):
        array = F.reshape(array, shape)

    return array
    

cdef object getitem(object self, object key):
    """Return self[key]."""
    assert isinstance(self, (NdArray, Variable))

    if isinstance(key, (list, np.ndarray, NdArray, Variable)):
        return advanced_getitem(self, [key])

    elif isinstance(key, (slice, int, type(Ellipsis), type(None))):
        return basic_getitem(self, [key])

    elif isinstance(key, tuple):
        sequence_types = (tuple, list, np.ndarray, NdArray, Variable)
        if any(isinstance(k, sequence_types) for k in key):
            return advanced_getitem(self, key)
        else:
            return basic_getitem(self, list(key))

    else:
        raise TypeError("invalid index {}".format(repr(key)))


def broadcast(x, shape):
    import nnabla.functions as F
    new_shape = (len(shape) - len(x.shape)) * (1,) + x.shape
    return F.broadcast(F.reshape(x, new_shape), shape=shape)


def advanced_setitem(self, key, value):
    """Advanced numpy-like indexing where the `key` object contains integer
    or boolean arrays. Sets the sub-shape produced by indexing `self` to
    the, potentially broadcast, `value` elements.
    """
    import nnabla.functions as F

    # set idx to key without np.newaxis
    idx = [k for k in key if k is not None]
    newaxes = list(1 if k is None else 0 for k in key)

    if sum(newaxes) > 0:
        subax = newaxes.index(0)
        shape = value.shape[subax:subax + 1] + value.shape[sum(newaxes) + 1:]
        value = F.reshape(value, shape)

    if len(idx) == 1 and isinstance(idx[0], (list, tuple)):
        a = np.array(idx[0])
        if a.dtype == np.bool:
            idx[0] = a

    if len(idx) == 1 and is_numpy_array(idx[0]) and idx[0].dtype == np.bool:
        indices = np.vstack(np.nonzero(idx[0]))

    elif all(isinstance(k, (NdArray, Variable)) for k in idx):
        indices = F.stack(*idx)

    elif all(isinstance(k, (int, tuple, list, np.ndarray)) for k in idx):
        indices = np.array(idx)

    else:
        indices = ', '.join(repr(val) for val in key)
        message = "mixed advanced indexing: {}".format(indices)
        raise NotImplementedError(message)

    upshape = indices.shape[1:] + self.shape[indices.shape[0]:]
    updates = broadcast(value, upshape) if value.shape != upshape else value
    F.scatter_nd(updates, indices, out=self)


def basic_setitem(self, key, value):
    """Basic numpy-like indexing where the `key` object contains only slices,
    integers, Ellipsis, or None (numpy.newaxis). Sets the sub-shape produced
    by indexing `self` to the, potentially broadcast, `value` elements.
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
    if sys.version_info.major < 3:
        stop = tuple(None if v == sys.maxsize else v for v in stop)
    index = F.reshape(F.arange(0, self.size), self.shape)
    index = F.slice(index, start, stop, step)

    count = itertools.count()
    shape = []
    for k in key:
        if isinstance(k, int):
            next(count)  # integer axis gets removed in basic indexing
        else:
            shape.append(1 if k is None else index.shape[next(count)])

    if index.shape != tuple(shape):
        index = F.reshape(index, shape)
    
    if value.shape != index.shape:
        value = broadcast(value, index.shape)
    
    value_flat = F.reshape(value, (value.size,))
    index_flat = F.reshape(index, (1, index.size))
    self_flat = F.reshape(self, (self.size,))
    F.scatter_nd(value_flat, index_flat, out=self_flat)


cdef object setitem(object self, object key, object value):
    """Perform self[key] = value.
    """
    assert isinstance(self, (NdArray, Variable))

    if not isinstance(value, (NdArray, Variable)):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        value = NdArray.from_numpy_array(value)

    if isinstance(key, (list, np.ndarray, NdArray, Variable)):
        return advanced_setitem(self, [key], value)

    elif isinstance(key, (slice, int, type(Ellipsis), type(None))):
        return basic_setitem(self, [key], value)

    elif isinstance(key, tuple):
        sequence_types = (tuple, list, np.ndarray, NdArray, Variable)
        if any(isinstance(k, sequence_types) for k in key):
            return advanced_setitem(self, key, value)
        else:
            return basic_setitem(self, list(key), value)

    else:
        raise TypeError("invalid index {}".format(repr(key)))
