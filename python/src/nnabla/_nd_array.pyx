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
from libcpp.algorithm cimport copy
from libcpp.memory cimport make_shared
from libc.stdint cimport intptr_t
from cpython cimport PyObject, Py_INCREF

from _nd_array cimport *
from _variable cimport *
from _array cimport *

# Numpy
import numpy as np
cimport numpy as np
np.import_array()


cdef class NdArray:
    """
    :class:`nnabla._nd_array.NdArray` is a device-agnostic data container for multi-dimensional arrays (tensors).
    :class:`nnabla._nd_array.NdArray` can also implictly handle data transfers across different devices (e.g. CPU to CUDA GPU, CUDA GPU to CPU).
    See `Python API Tutorial <http://nnabla.readthedocs.io/en/latest/python/tutorial/python_api.html>`_ for more details.


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

        Retuns: int

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
        import nnabla as nn
        ctx_ = nn.context()
        if ctx is not None:
            ctx_ = ctx
        cdef int type_num = np.dtype(dtype).num
        cdef CContext cctx = <CContext ?> ctx_
        with nogil:
            self.arrp.cast(< dtypes > type_num, cctx)
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
        cdef int type_num
        cdef vector[np.npy_intp] shape
        cdef Shape_t shape_base
        cdef CArray * arr
        import nnabla as nn
        ctx = nn.context()
        cdef CContext cctx = <CContext > ctx
        try:
            type_num = <int > self.arrp.array().get().dtype()
        except:
            type_num = np.dtype(np.float32).num
        shape.resize(self.arrp.ndim())
        shape_base = self.arrp.shape()
        copy(shape_base.begin(), shape_base.end(), shape.begin())
        with nogil:
            arr = <CArray * > (self.arrp.cast( < dtypes > type_num, cctx))
        cdef np.ndarray ndarray = np.PyArray_SimpleNewFromData(
            shape.size(), shape.data(), type_num, arr.pointer())
        ndarray.base = <PyObject * > self
        Py_INCREF(self)
        return ndarray

    @data.setter
    def data(self, value):
        self.data[...] = value

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
        """
        This function simply returns itself.
        Implements the unary plus operator, ``+A``.

        Returns: :class:`nnabla.NdArray`
        """
        return self

    def __neg__(self):
        """
        Element-wise negation.
        Implements the unary negation expression ``-A`` .

        Returns: :class:`nnabla.NdArray`

        """
        import nnabla.functions as F
        return F.mul_scalar(self, -1)

    def __add__(x, y):
        """
        Element-wise addition.

        Implements the addition operator expression ``x + y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.Variable` or
        :obj:`~nnabla.NdArray`, :func:`~nnabla.functions.add2`` is
        internally called.
        When one of ``x`` and ``y`` is a scalar,
        :func:`~nnabla.functions.add_scalar` is called.

        Args:
            x (float or ~nnabla.Variable or ~nnabla.NdArray): Left operand.
            y (float or ~nnabla.Variable or ~nnabla.NdArray): Right operand.

        Returns: :class:`~nnabla.Variable` or :class:`~nnabla.NdArray`.

        """
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            if isinstance(y, (NdArray, Variable)):
                return F.add2(x, y)
            else:
                return F.add_scalar(x, y)
        else:
            if isinstance(y, (NdArray, Variable)):
                return F.add_scalar(y, x)
            else:
                return x + y

    def __sub__(x, y):
        """
        Element-wise subtraction.

        Implements the subtraction operator expression ``x - y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.Variable` or
        :obj:`~nnabla.NdArray`, :func:`~nnabla.functions.sub2`` is
        internally called.
        When ``y`` is a scalar, :func:`~nnabla.functions.add_scalar`(x, -y) is
        called. When ``x`` is a scalar,
        :func:`~nnabla.functions.r_sub_scalar`(y, x) is called.

        Args:
            x (float or ~nnabla.Variable or ~nnabla.NdArray): Left operand.
            y (float or ~nnabla.Variable or ~nnabla.NdArray): Right operand.

        Returns: :class:`~nnabla.Variable` or :class:`~nnabla.NdArray`.

        """
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            if isinstance(y, (NdArray, Variable)):
                return F.sub2(x, y)
            else:
                return F.add_scalar(x, -y)
        else:
            if isinstance(y, (NdArray, Variable)):
                return F.r_sub_scalar(y, x)
            else:
                return x - y

    def __mul__(x, y):
        """
        Element-wise multiplication.

        Implements the multiplication operator expression ``x * y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.Variable` or
        :obj:`~nnabla.NdArray`, :func:`~nnabla.functions.mul2`` is
        internally called.
        When one of ``x`` and ``y`` is a scalar,
        :func:`~nnabla.functions.mul_scalar` is called.

        Args:
            x (float or ~nnabla.Variable or ~nnabla.NdArray): Left operand.
            y (float or ~nnabla.Variable or ~nnabla.NdArray): Right operand.

        Returns: :class:`~nnabla.Variable` or :class:`~nnabla.NdArray`.

        """
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            if isinstance(y, (NdArray, Variable)):
                return F.mul2(x, y)
            else:
                return F.mul_scalar(x, y)
        else:
            if isinstance(y, (NdArray, Variable)):
                return F.mul_scalar(y, x)
            else:
                return x * y

    def __truediv__(x, y):
        """
        Element-wise division.

        Implements the division operator expression ``x / y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.Variable` or
        :obj:`~nnabla.NdArray`, :func:`~nnabla.functions.div2`` is
        internally called.
        When ``y`` is a scalar, :func:`~nnabla.functions.div_scalar`(x, y) is
        called. When ``x`` is a scalar,
        :func:`~nnabla.functions.r_div_scalar`(y, x) is called.

        Args:
            x (float or ~nnabla.Variable or ~nnabla.NdArray): Left operand.
            y (float or ~nnabla.Variable or ~nnabla.NdArray): Right operand.

        Returns: :class:`~nnabla.Variable` or :class:`~nnabla.NdArray`.

        """
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            if isinstance(y, (NdArray, Variable)):
                return F.div2(x, y)
            else:
                return F.mul_scalar(x, 1 / y)
        else:
            if isinstance(y, (NdArray, Variable)):
                return F.r_div_scalar(y, x)
            else:
                return x / y

    def __div__(x, y):
        """
        Element-wise division.

        Implements the division operator expression ``x / y``.
        When both of ``x`` and ``y`` are either :obj:`~nnabla.Variable` or
        :obj:`~nnabla.NdArray`, :func:`~nnabla.functions.div2`` is
        internally called.
        When ``y`` is a scalar, :func:`~nnabla.functions.div_scalar`(x, y) is
        called. When ``x`` is a scalar,
        :func:`~nnabla.functions.r_div_scalar`(y, x) is called.

        Args:
            x (float or ~nnabla.Variable or ~nnabla.NdArray): Left operand.
            y (float or ~nnabla.Variable or ~nnabla.NdArray): Right operand.

        Returns: :class:`~nnabla.Variable` or :class:`~nnabla.NdArray`.

        """
        import nnabla.functions as F
        if isinstance(x, (NdArray, Variable)):
            if isinstance(y, (NdArray, Variable)):
                return F.div2(x, y)
            else:
                return F.mul_scalar(x, 1 / y)
        else:
            if isinstance(y, (NdArray, Variable)):
                return F.r_div_scalar(y, x)
            else:
                return x / y

    def __pow__(x, y, z):
        """
        Element-wise power function.

        Implements the power operator expression ``x ** y``,
        optionaly ``x ** y % z`` (but not implemented).
        When both of ``x`` and ``y`` are either :obj:`~nnabla.Variable` or
        :obj:`~nnabla.NdArray`, :func:`~nnabla.functions.pow2`` is
        internally called.
        When ``y`` is a scalar, :func:`~nnabla.functions.pow_scalar`(x, y) is
        called. When ``x`` is a scalar,
        :func:`~nnabla.functions.r_pow_scalar`(y, x) is called.

        Args:
            x (float or ~nnabla.Variable or ~nnabla.NdArray): Left operand.
            y (float or ~nnabla.Variable or ~nnabla.NdArray): Right operand.
            z (float or ~nnabla.Variable or ~nnabla.NdArray): Modulo (optional).

        Returns: :class:`~nnabla.Variable` or :class:`~nnabla.NdArray`.

        """
        import nnabla.functions as F
        if z is not None:
            return NotImplemented
        if isinstance(x, (NdArray, Variable)):
            if isinstance(y, (NdArray, Variable)):
                return F.pow2(x, y)
            else:
                return F.pow_scalar(x, y)
        else:
            if isinstance(y, (NdArray, Variable)):
                return F.r_pow_scalar(y, x)
            else:
                return x ** y
