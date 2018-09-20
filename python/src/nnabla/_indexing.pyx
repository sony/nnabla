from _variable cimport Variable
from _nd_array cimport NdArray

# Numpy
import numpy as np
cimport numpy as np
np.import_array()


def _align_key_and_shape(key, shape):
    # TODO: refactor
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
        return key, _shape
    elif len(key) < len(shape):
        _key = []
        _shape = []
        for i in range(len(shape)):
            if i < len(key):
                _key.append(key[i])
                _shape.append(shape[i])
            else:
                _key.append(slice(None))
                _shape.append(shape[i])
        return _key, _shape
    elif len(key) > len(shape):
        raise IndexError("too many indices for array")
    else:
        return key, shape


def _force_to_same_len_list(object key, shape):
    reshape_hint = []
    keys = []
    if isinstance(key, int):
        keys.append(slice(key, key + 1, 1))
        # Do NOT add k to shape hint
        for i, s in enumerate(shape[1:]):
            keys.append(slice(0, s, 1))
            reshape_hint.append(i + 1)
    elif isinstance(key, slice):
        _slice = key
        keys.append(_slice)
        reshape_hint.append(0)
        for i, s in enumerate(shape[1:]):
            keys.append(slice(0, s, 1))
            reshape_hint.append(i + 1)
    elif key == Ellipsis:
        for i, s in enumerate(shape):
            keys.append(slice(0, s, 1))
            reshape_hint.append(i)
    elif key == np.newaxis:
        reshape_hint.append(np.newaxis)
        for i, s in enumerate(shape):
            keys.append(slice(0, s, 1))
            reshape_hint.append(i)
    elif isinstance(key, tuple):
        key, shape = _align_key_and_shape(key, shape)
        cnt = 0
        for i, ks in enumerate(zip(key, shape)):
            k, s = ks
            if isinstance(k, int):
                keys.append(slice(k, k + 1, 1))
                # Do NOT add k to shape hint
            elif isinstance(k, slice):
                _slice = k
                keys.append(_slice)
                reshape_hint.append(i - cnt)
            elif k == np.newaxis:
                reshape_hint.append(np.newaxis)
                cnt += 1
    else:
        raise IndexError(
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)")

    return keys, reshape_hint  # list of slice, list of integer


cdef object getitem(object self, object key):
    """Basic numpy-like indexing.
    Returns: :class:`nnabla.Variable` or :class:`nnabla.NdArray`
    """

    import nnabla.functions as F
    # Get shape
    if isinstance(self, NdArray):
        shape = (< NdArray > self).shape
    elif isinstance(self, Variable):
        shape = (< Variable > self).shape
    else:
        raise IndexError("self should be NdArray or Variable")

    # TODO: Advanced Indexing
    start, stop, step = [], [], []
    keys, reshape_hint = _force_to_same_len_list(key, shape)

    # Slice first
    for key in keys:
        start.append(key.start)
        stop.append(key.stop)
        step.append(key.step)
    x_sliced = F.slice(self, start, stop, step)

    # Reshape
    shape = x_sliced.shape
    shape_new = []
    new_axis_cnt = 0
    for rh in reshape_hint:
        if rh == np.newaxis:
            shape_new.append(1)
        else:
            shape_new.append(shape[rh])

    return F.reshape(x_sliced, shape_new)
