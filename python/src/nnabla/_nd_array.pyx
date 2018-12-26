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

from __future__ import division
from libcpp cimport bool as cpp_bool
from libcpp.algorithm cimport copy
from libcpp.memory cimport make_shared, shared_ptr
from libc.stdint cimport intptr_t
from cpython cimport PyObject, Py_INCREF

from _nd_array cimport *
from _variable cimport *
from _array cimport *
cimport _arithmetic_ops as AOP
cimport _indexing as IDX

# Numpy
import numpy as np
cimport numpy as np
np.import_array()

# Older cython doesn't expose const_pointer_cast in <memory>
cdef extern from "<memory>" namespace "std" nogil:
    cdef shared_ptr[T] const_pointer_cast[T, U](const shared_ptr[U] & )


cdef c_get_numpy_array(CNdArray * arrp, vector[np.npy_intp] & shape,
                       int type_num, CContext cctx):
    cdef shared_ptr[const CArray] arr
    with nogil:
        arr = <shared_ptr[const CArray] > (arrp.get_sp( < dtypes > type_num, cctx))
    cdef np.ndarray ndarray = np.PyArray_SimpleNewFromData(
        shape.size(), shape.data(), type_num, < void*>(arr.get().const_pointer()))
    ndarray.flags.writeable = False
    pyarr = Array.create(arr)
    ndarray.base = <PyObject * > pyarr
    Py_INCREF(pyarr)
    return ndarray


cdef c_cast_numpy_array(CNdArray * arrp, vector[np.npy_intp] & shape,
                        int type_num, CContext cctx, cpp_bool write_only):
    cdef ArrayPtr arr
    with nogil:
        arr = <ArrayPtr > (arrp.cast_sp(< dtypes > type_num, cctx, write_only))
    cdef np.ndarray ndarray = np.PyArray_SimpleNewFromData(
        shape.size(), shape.data(), type_num, arr.get().pointer())
    cdef shared_ptr[const CArray] carr = < shared_ptr[const CArray] > const_pointer_cast[ConstArray, CArray](arr)
    pyarr = Array.create(carr)
    ndarray.base = <PyObject * > pyarr
    Py_INCREF(pyarr)
    return ndarray


cdef c_as_numpy_array(CNdArray * arrp, str mode):
    cdef int type_num
    cdef vector[np.npy_intp] shape
    cdef Shape_t shape_base
    from nnabla_ext.cpu import context
    ctx = context()
    cdef CContext cctx = <CContext > ctx

    # Getting current data type
    try:
        type_num = <int > arrp.array().get().dtype()
    except:
        type_num = np.dtype(np.float32).num

    # Create numpy shape array
    shape.resize(arrp.ndim())
    shape_base = arrp.shape()
    copy(shape_base.begin(), shape_base.end(), shape.begin())

    # Convert to numpy array with flags depending on the mode option.
    if mode == 'r':
        return c_get_numpy_array(arrp, shape, type_num, cctx)
    else:
        assert mode in ('w', 'rw'), 'Invalid mode is given: "%s"' % mode
        return c_cast_numpy_array(arrp, shape, type_num, cctx, mode == 'w')


cdef class NdArray:
    """
    :class:`nnabla._nd_array.NdArray` is a device-agnostic data container for multi-dimensional arrays (tensors).
    :class:`nnabla._nd_array.NdArray` can also implicitly handle data transfers across different devices (e.g. CPU to CUDA GPU, CUDA GPU to CPU).
    See `Python API Tutorial <http://nnabla.readthedocs.io/en/latest/python/tutorial/python_api.html>`_ for more details.

    :class:`~nnabla.NdArray` overrides some arithmetic operators
    (``+``, ``-``, ``*``, ``/``, ``**``). Operands can be either a scalar number,
    :class:`~nnabla.NdArray` or :class:`~nnabla.Variable`.
    An arithmetic operation containing :class:`~nnabla.NdArray` returns
    :class:`~nnabla.NdArray` which stores the output of the computation
    immediately invoked.
    Also, inplace arithmetic operations
    (``+=``, ``-=``, ``*=``, ``/=``, ``**=``) are implemented. Note that ``=``
    doesn't perform inplace substitution but just replaces the object
    reference. Instead, you can use :func:`~nnabla.NdArray.copy_from` for
    inplace substitution.

    Args:
        shape (tuple or int): Shape of tuple.

    """

    @staticmethod
    cdef object create(NdArrayPtr arr):
        a = NdArray()
        a.arr = arr
        a.arrp = arr.get()
        return a

    @staticmethod
    def from_numpy_array(nparr):
        """Create a NdArray object from Numpy array data.

        The data is initialized with the given Numpy array.

        Args:
            nparr (~numpy.ndarray): Numpy multi-dimensional array.

        Returns: ~nnabla._nd_array.NdArray

        """
        assert isinstance(nparr, np.ndarray)
        a = NdArray(nparr.shape)
        a.cast(nparr.dtype)
        a.data = nparr
        return a

    def __init__(self, shape=tuple()):
        cdef int i
        cdef Shape_t cshape
        cdef int size = len(shape)
        cshape.resize(size)
        for i in range(size):
            cshape[i] = shape[i]
        self.arr = make_shared[CNdArray](cshape)
        self.arrp = self.arr.get()

    def __repr__(self):
        return "<NdArray({}) at {}>".format(
            self.shape, hex(id(self)))

    def __richcmp__(self, other, int op):
        '''Overrides comparison operators ``==`` and ``!=``.

        Compare the addresses of their C++ objects.
        '''
        if op == 2:
            try:
                return (< NdArray > self).arrp == ( < NdArray ?> other).arrp
            except:
                return False
        elif op == 3:
            return not self.__richcmp__(other, 2)
        return False

    def __hash__(self):
        '''Returns hash of the integer address of holding C++ object.
        '''
        return hash(< intptr_t > (( < NdArray > self).arrp))

    @property
    def shape(self):
        """Shape of the N-d array.

        Returns: tuple of int

        """
        return tuple(self.arrp.shape())

    @property
    def size(self):
        """Total size of the N-d array.

        Returns: int

        """
        return self.arrp.size(-1)

    def size_from_axis(self, axis=-1):
        """
        Gets the size followed by the provided axis.

        Example:

            .. code-block:: python

                a = nnabla.NdArray([10,9])
                a.size_from_axis()
                # ==> 90
                a.size_from_axis(0)
                # ==> 90
                a.size_from_axis(1)
                # ==> 9
                a.size_from_axis(2)
                # ==> 1

        Args:
            axis (:obj:`int`, optional): -1 as default

        Returns:
            :obj:`int`

        """
        return self.arrp.size(axis)

    @property
    def strides(self):
        """Strides.

        Returns: tuple of int

        """
        return self.arrp.strides()

    @property
    def ndim(self):
        """Number of dimensions.

        Returns: int 

        """
        return self.arrp.ndim()

    def cast(self, dtype, ctx=None):
        """
        In-place cast of data type of the NdArray. It returns the reference
        values as a numpy.ndarray only if optional parameter ctx is not given,
        None otherwise.

        Args: 
            dtype (:obj:`numpy.dtype`):  Numpy Data type.
            ctx (:obj:`nnabla.Context`, optional): Context descriptor.

        Returns:
            :obj:`numpy.array` if ``ctx`` is None, otherwise nothing.
        """
        from nnabla_ext.cpu import context
        ctx_ = context()
        if ctx is not None:
            ctx_ = ctx
        cdef int type_num = np.dtype(dtype).num
        cdef CContext cctx = <CContext ?> ctx_
        with nogil:
            self.arrp.cast(< dtypes > type_num, cctx, False)
        if ctx is None:
            return self.data

    @property
    def data(self):
        """
        Returns the values held by this array as a :class:`numpy.ndarray`.
        Note that only the references are returned, and the values are not copied. Therefore,
        modifying the returned :class:`nnabla._nd_array.NdArray` will affect the data contained inside the
        NNabla array.
        This method can also be called as a setter.
        Note that this may implicitly invoke a data transfer from device arrays to the CPU.

        Args:
            value (:obj:`numpy.ndarray`)

        Returns: :obj:`numpy.ndarray`

        """
        return c_as_numpy_array(self.arrp, 'rw')

    @data.setter
    def data(self, value):
        self.data[...] = value

    def get_data(self, str mode='rw'):
        '''
        Returns the values held by this array as a :class:`numpy.ndarray`
        with a specified mode.

        Args:
            mode (str): Computation becomes more efficient if right one is chosen.
                * 'r': Read-only access.
                * 'w': Write-only access.
                * 'rw': You can both read and write.

        See :function:`nnabla._nd_array.NdArray.data for more details.

        '''
        return c_as_numpy_array(self.arrp, mode)

    def zero(self):
        """
        Fill all of the elements with 0.

        Note: This method is lazily evaluated. It is evaluated during the forward or
        backward propagation.

        """
        self.arrp.zero()

    def fill(self, value):
        """
        Fill all of the elements with the provided scalar value.

        Note: This method is lazily evaluated. It is evaluated during the forward or
        backward propagation.

        Args:
            value (int, float): The value filled with. 

        """
        self.arrp.fill(value)

    @property
    def dtype(self):
        """
        Get dtype.

        Returns: :obj:`numpy.dtype`

        """
        cdef int type_num
        type_num = <int > self.arrp.array().get().dtype()
        return np.dtype(np.PyArray_TypeObjectFromType(type_num))

    def __pos__(self):
        return AOP.pos(self)

    def __neg__(self):
        return AOP.neg(self)

    def __add__(x, y):
        return AOP.add(x, y)

    def __sub__(x, y):
        return AOP.sub(x, y)

    def __mul__(x, y):
        return AOP.mul(x, y)

    def __truediv__(x, y):
        return AOP.truediv(x, y)

    def __div__(x, y):
        return AOP.div(x, y)

    def __pow__(x, y, z):
        return AOP.pow(x, y, z)

    def __iadd__(self, x):
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            F.add2(self, x, outputs=[self])
        else:
            F.add_scalar(self, x, outputs=[self])
        return self

    def __isub__(self, x):
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            F.sub2(self, x, outputs=[self])
        else:
            F.add_scalar(self, -x, outputs=[self])
        return self

    def __imul__(self, x):
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            F.mul2(self, x, outputs=[self])
        else:
            F.mul_scalar(self, x, outputs=[self])
        return self

    def __idiv__(self, x):
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            F.div2(self, x, outputs=[self])
        else:
            F.mul_scalar(self, 1. / x, outputs=[self])
        return self

    def __itruediv__(self, x):
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            F.div2(self, x, outputs=[self])
        else:
            F.mul_scalar(self, 1. / x, outputs=[self])
        return self

    def __ipow__(self, x):
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            F.pow2(self, x, outputs=[self])
        else:
            F.pow_scalar(self, x, outputs=[self])
        return self

    def copy_from(self, NdArray arr):
        """
        Copy values from another NdArray object.

        It returns the caller object itself.
        :func:`nnabla.functions.identity` is called internally to copy values.

        Args:
            arr (~nnabla.NdArray): Values will be copied to the caller object.
                The shape of ``arr``` must be same as the caller object.

        Returns:
            :obj:`nnabla.NdArray`

        """
        import nnabla.functions as F
        F.identity(arr, outputs=[self])
        return self

    def __getitem__(self, key):
        return IDX.getitem(self, key)
