from _variable cimport Variable
from _nd_array cimport NdArray

# Numpy
import numpy as np
cimport numpy as np
np.import_array()


def _fill_slice(_slice, s):
    _slice = slice(_slice.start if _slice.start is not None else 0,
                   _slice.stop if _slice.stop is not None else s,
                   _slice.step if _slice.step is not None else 1)
    return _slice


def _infer_size(_slice):
    return (_slice.stop - _slice.start) // _slice.step + (0 if _slice.step == 1 else 1)


def _align_key_and_shape(key, shape):
    if Ellipsis in key:  # it must be a single ellipsis ('...')
        i = [i for i, k in enumerate(key) if k == Ellipsis][0]
        n_newaxis = len([k for k in key if k == np.newaxis])
        n = len(shape) - (len(key) - 1) + n_newaxis
        k0 = key[:i] + tuple([slice(None)] * n)
        key = k0 if i == len(key) - 1 else k0 + key[i + 1:]

        _shape = []
        cnt = 0
        for k in key:
            if k == np.newaxis:
                _shape.append(None)  # dummy
                continue
            _shape.append(shape[cnt])
            cnt += 1
    else:
        _shape = shape
    return key, _shape


def _force_to_same_len_list(object key, shape):
    reshape = []
    keys = []
    if isinstance(key, int):
        keys.append(slice(key, key + 1, 1))
        for s in shape[1:]:
            keys.append(slice(0, s, 1))
            reshape.append(s)
    elif isinstance(key, slice):
        _slice = _fill_slice(key, shape[0])
        keys.append(_slice)
        s = _infer_size(_slice)
        reshape.append(s)
        for s in shape[1:]:
            keys.append(slice(0, s, 1))
            reshape.append(s)
    elif key == Ellipsis:
        for s in shape:
            keys.append(slice(0, s, 1))
            reshape.append(s)
    elif key == np.newaxis:  # np.newaxis
        reshape.append(1)
        for s in shape:
            keys.append(slice(0, s, 1))
            reshape.append(s)
    elif isinstance(key, tuple):
        key, shape = _align_key_and_shape(key, shape)
        for k, s in zip(key, shape):
            if isinstance(k, int):
                keys.append(slice(k, k + 1, 1))
            elif isinstance(k, slice):
                _slice = _fill_slice(k, s)
                keys.append(_slice)
                s = _infer_size(_slice)
                reshape.append(s)
            elif k == np.newaxis:  # np.newaxis
                reshape.append(1)
    else:
        raise IndexError(
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)")

    return keys, reshape  # list of slice, list of integer


cdef object getitem(object self, object key):
    """Basic numpy-like indexing.
    Returns: :class:`nnabla.Variable` or :class:`nnabla.NdArray`
    """

    import nnabla.functions as F
    # Get shape
    if isinstance(self, NdArray):
        shape = ( < NdArray > self).shape
    elif isinstance(self, Variable):
        shape = ( < Variable > self).shape
    else:
        raise ValueError("self should be NdArray or Variable")

    # TODO: Negative Indexing
    # TODO: Advanced Indexing
    # Basic Slicing and Indexing without negative index

    start, stop, step = [], [], []
    keys, reshape = _force_to_same_len_list(key, shape)
    for key in keys:
        start.append(key.start)
        stop.append(key.stop)
        step.append(key.step)

    return F.reshape(F.slice(self, start, stop, step), reshape)
