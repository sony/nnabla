from _variable cimport Variable
from _nd_array cimport NdArray

import numpy as np
import itertools
import sys


def is_python_boolean_sequence(idx):
    return (isinstance(idx, (tuple, list)) and len(idx) > 0
            and all(isinstance(x, bool) for x in idx))


def is_numpy_boolean_ndarray(idx):
    return isinstance(idx, np.ndarray) and idx.dtype == np.bool


def is_nnabla_boolean_ndarray(idx):
    return isinstance(idx, NdArray) and idx.dtype == np.bool


def is_nnabla_boolean_variable(idx):
    return isinstance(idx, Variable) and idx.data.dtype == np.bool


def broadcast(x, shape):
    import nnabla.functions as F
    new_shape = (len(shape) - len(x.shape)) * (1,) + x.shape
    return F.broadcast(F.reshape(x, new_shape), shape=shape)


def basic_getsetitem_prepare(self, key):
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

    return key, idx, start, stop, step


def basic_getsetitem_result_shape(self, key, array):
    count = itertools.count()
    shape = []

    for k in key:
        if isinstance(k, int):
            next(count)  # integer axis gets removed in basic indexing
        else:
            shape.append(1 if k is None else array.shape[next(count)])

    return tuple(shape)


def basic_getitem(self, key):
    """Basic numpy-like indexing where the `key` object contains only slices,
    integers, Ellipsis or None (numpy.newaxis). Returns the indexed sub-space
    of `self` as a :class:`nnabla.Variable` or :class:`nnabla.NdArray`
    depending on the type of `self`.
    """
    import nnabla.functions as F

    key, idx, start, stop, step = basic_getsetitem_prepare(self, key)

    sliced_array = F.slice(self, start, stop, step)
    sliced_shape = basic_getsetitem_result_shape(self, key, sliced_array)
    
    if sliced_array.shape != sliced_shape:
        sliced_array = F.reshape(sliced_array, sliced_shape)
    
    return sliced_array


def basic_setitem(self, key, value):
    """Basic numpy-like indexing where the `key` object contains only slices,
    integers, Ellipsis, or None (numpy.newaxis). Sets the sub-shape produced
    by indexing `self` to the, potentially broadcast, `value` elements.
    """
    import nnabla.functions as F

    key, idx, start, stop, step = basic_getsetitem_prepare(self, key)

    array = self.data if isinstance(self, Variable) else self
    sliced_array = F.slice(array, start, stop, step)
    sliced_shape = basic_getsetitem_result_shape(self, key, sliced_array)

    if value.shape != sliced_shape:
        value = broadcast(value, sliced_shape)

    indices = (np.arange(self.shape[ax])[ix] for ax, ix in enumerate(idx))
    indices = np.array(np.meshgrid(*indices, indexing='ij'))

    if value.shape != indices.shape[1:]:
        value = F.reshape(value, indices.shape[1:])

    return F.scatter_nd(value, indices, out=self)


def advanced_getitem(self, key):
    """Advanced numpy-like indexing where the `key` object contains
    integer or boolean arrays. Returns the indexed sub-space of `self`
    as a :class:`nnabla.Variable` or :class:`nnabla.NdArray` depending
    on the type of `self`.
    """
    import nnabla.functions as F
    idx = [val for val in key if val is not np.newaxis]
    newaxes = list(1 if k is None else 0 for k in key)

    # Note: For a boolean index array the result shape depends on the truth
    # value of the index data elements and must be determined as part of the
    # computation graph setup phase. Thus, even if the index is a Variable
    # or NdArray, the data is evaluated here through np.nonzero() and not
    # during forward.

    if len(idx) == 1 and is_nnabla_boolean_ndarray(idx[0]):
        result = F.gather_nd(self, np.vstack(np.nonzero(idx[0].data)))

    elif len(idx) == 1 and is_nnabla_boolean_variable(idx[0]):
        result = F.gather_nd(self, np.vstack(np.nonzero(idx[0].d)))

    elif len(idx) == 1 and is_python_boolean_sequence(idx[0]):
        result = F.gather_nd(self, np.vstack(np.nonzero(idx[0])))

    elif len(idx) == 1 and is_numpy_boolean_ndarray(idx[0]):
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

    if len(idx) == 1 and is_nnabla_boolean_ndarray(idx[0]):
        indices = np.vstack(np.nonzero(idx[0].data))

    elif len(idx) == 1 and is_nnabla_boolean_variable(idx[0]):
        indices = np.vstack(np.nonzero(idx[0].d))

    elif len(idx) == 1 and is_python_boolean_sequence(idx[0]):
        indices = np.vstack(np.nonzero(idx[0]))

    elif len(idx) == 1 and is_numpy_boolean_ndarray(idx[0]):
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
    return F.scatter_nd(updates, indices, out=self)


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
