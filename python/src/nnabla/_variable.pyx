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

from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport int64_t, intptr_t
from libcpp.memory cimport make_shared, shared_ptr
from cpython cimport PyObject, Py_INCREF
cimport _variable
from _variable cimport CVariable, CContext, Shape_t, dtypes
cimport function
from function cimport CgFunction

# Numpy
import numpy as np
cimport numpy as np
np.import_array()

cimport _arithmetic_ops as AOP


ctypedef void * voidp


cdef class Context:

    """
    Context is used to specify the computation engine (cpu, cuda, cuda.cudnn etc.) which the
    function operator modules and optimizer modules shall be ran on.
    The context can be set for each function, as well as set globally with functions
    listed in the :meth:`context-specifier`.

    Args:
        backend (list of str): 'cpu', 'cuda', 'cuda.cudnn' etc.
        array_class (str): str, 'CpuArray', 'CpuCachedArray', 'CudaArray', 'CudaCachedArray' etc.
        device_id (str): str, default '0'

    """

    def __init__(self, backend=None, array_class='',
                 device_id='0'):
        if backend is None:
            backend = ['cpu:float']
        for b in backend:
            self.backend_.push_back(b)
        self.array_class = array_class
        self.device_id = device_id

    @property
    def backend(self):
        ret = []
        for b in self.backend_:
            ret.append(b)
        return ret

    @backend.setter
    def backend(self, backends):
        self.backend_.resize(len(backends))
        for i, b in enumerate(backends):
            self.backend_[i] = b

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return "Context(backend='{}', array_class='{}'"\
            ", device_id='{}')".format(
                self.backend, self.array_class,
                self.device_id)

    def __str__(self):
        return repr(self)


cdef class Variable:
    """
    :class:`nnabla.Variable` is used to construct computation graphs (neural networks) together
    with functions in :ref:`functions`
    and :ref:`parametric-functions` .
    It also provides a method to execute forward and backward
    propagation of the network.
    The :class:`nnabla.Variable` class holds:

    * Reference to the parent function in a
      computation graph. This provides traceability of all connections in the computation graph.
    * Both data and error
      signal (gradient) containers as :class:`nnabla._nd_array.NdArray` s.
    * Some additional information of the computation graph.

    :class:`~nnabla.Variable` overrides some arithmetic operators
    (``+``, ``-``, ``*``, ``/``, ``**``). Operands can be either a scalar number,
    :class:`~nnabla.NdArray` or :class:`~nnabla.Variable`.
    If :class:`~nnabla.NdArray` is given as either of left or right operand,
    the arithmetic operation returns an :class:`~nnabla.NdArray` which stores the
    output of the computation immediately invoked. Otherwise, it returns
    :class:`~nnabla.Variable` holds the graph connection. The computation
    is invoked immediately when :function:`nnabla.auto_forward`
    or :function:`nnabla.set_auto_forward(True)` is used.

    See also:
        `Python API Tutorial
        <http://nnabla.readthedocs.io/en/latest/python/tutorial/python_api.html>`_.

    Args:
        shape (Iterable of int): Shape of variable.
        need_grad (bool): Flag for backprop or not.

    """

    def __cinit__(self, Shape_t shape=[], bint need_grad=False, info=None):
        self.info = info
        self.var = make_shared[CgVariable](shape, need_grad)
        self.varp = self.var.get()

    @staticmethod
    cdef create_from_cvariable(shared_ptr[CVariable] varsp):
        cdef shared_ptr[CgVariable] v_sp = make_shared[CgVariable](varsp)
        var = Variable()
        var.var = v_sp
        var.varp = v_sp.get()
        return var

    @staticmethod
    cdef create_from_cg_variable(CgVariablePtr cgv):
        var = Variable()
        var.var = cgv
        var.varp = cgv.get()
        return var

    @staticmethod
    def from_numpy_array(data, grad=None, need_grad=False):
        """Create a Variable object from Numpy array(s).

        The ``data`` is initialized with the given Numpy array, as well as
        ``grad`` if given.

        The shape is also determined by the given array.

        Args:
            data (~numpy.ndarray): Values copied to the ``data`` of the created
                Variable.
            grad (~numpy.ndarray): Values copied to the ``grad`` of the created
                Variable.
            need_grad (bool): Flag for backprop or not.

        Returns: ~nnabla.Variable

        """
        assert isinstance(data, np.ndarray)
        var = Variable(data.shape, need_grad)
        var.data.cast(data.dtype)
        var.d = data
        if grad is None:
            return var
        assert isinstance(grad, np.ndarray)
        assert data.shape == grad.shape
        var.grad.cast(grad.dtype)
        var.g = grad
        return var

    def __dealloc__(self):
        pass

    def __repr__(self):
        return "<Variable({}, need_grad={}) at {}>".format(
            self.shape, self.need_grad, hex(id(self)))

    def __richcmp__(self, other, int op):
        '''Overrides comparison operators ``==`` and ``!=``.

        Compare the addresses of their C++ objects.
        '''
        if op == 2:
            try:
                return (< Variable > self).varp == ( < Variable ?> other).varp
            except:
                return False
        elif op == 3:
            return not self.__richcmp__(other, 2)
        return False

    def __hash__(self):
        '''Returns hash of the integer address of holding C++ object.
        '''
        return hash(< intptr_t > (( < Variable > self).varp))

    @property
    def shape(self):
        """
        Gets the shape of the variable.


        Returns: tuple of :obj:`int`

        """
        return tuple(self.varp.variable().get().shape())

    @property
    def size(self):
        """
        Gets the size of the variable.

        Returns: :obj:`int`

        """
        return self.varp.variable().get().size(-1)

    @property
    def ndim(self):
        """
        Gets the number of dimensions of this variable.

        Returns: int

        """
        return self.varp.variable().get().ndim()

    def size_from_axis(self, axis=-1):
        """
        Gets the size followed by the provided axis.

        Example:

            .. code-block:: python

                a = nnabla.Variable([10,9])
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
        return self.varp.variable().get().size(axis)

    def reset_shape(self, shape, force=False):
        """Resizes the shape of the variable to a specified shape.

        Args:
            shape (Iterable of int): Target shape.
            force (bool): Flag to force reshape.

        Note:
            This method destructively changes the shape of the target variable. For safety, :func:`~nnabla.functions.reshape` should be used instead.

        Returns:
            None

        """
        self.varp.variable().get().reshape(shape, force)

    def reshape(self, shape, unlink=False):
        """Returns a new variable, where this variable is reshaped to a specified shape.

        Args:
          shape (Iterable of int): Target shape.
          unlink (bool): Unlink graph connection. Or, keep graph connection, i.e.
            the gradient will be backprop-ed to the original variable.

        Returns:
            :class:`~nnabla.Variable`

        """
        if unlink:
            var = Variable.create_from_cvariable(
                self.varp.variable().get().view(shape))
            return var
        from nnabla.functions import reshape
        return reshape(self, shape)

    @property
    def need_grad(self):
        """
        Gets or sets a boolean indicating whether backpropagation is performed at this variable. 

        Args:
            b (bool): Whether backpropagation is performed at this variable.

        Returns:
           bool: Whether this variable requires gradient or not.
        """
        return self.varp.variable().get().need_grad()

    @need_grad.setter
    def need_grad(self, b):
        cdef CgFunctionPtr parent = self.varp.parent()
        self.varp.variable().get().set_need_grad(b)
        # Reset need_grad flag of the parent function.
        if parent:
            parent.get().update_need_grad()

    @property
    def data(self):
        """Returns the data held by this variable, as a
        :class:`~nnabla._nd_array.NdArray`. This can also be used as a setter.

        Args:
            ndarray (~nnabla._nd_array.NdArray): NdArray object. Size must
                be the same as this Variable.

        Returns:
            :class:`~nnabla._nd_array.NdArray`
        """
        return NdArray.create(self.varp.variable().get().data())

    @data.setter
    def data(self, NdArray ndarray):
        self.varp.variable().get().set_data(ndarray.arr)

    @property
    def grad(self):
        """Returns the gradient held by this variable, as a
        :class:`~nnabla._nd_array.NdArray`. This can also be used as a setter.

        Args:
            ndarray (~nnabla._nd_array.NdArray): NdArray object. Size must
                be the same as this Variable.

        Returns:
            :class:`~nnabla._nd_array.NdArray`
        """
        return NdArray.create(self.varp.variable().get().grad())

    @grad.setter
    def grad(self, NdArray ndarray):
        self.varp.variable().get().set_grad(ndarray.arr)

    @property
    def d(self):
        """
        Returns the values held by this variable, as a :class:`numpy.ndarray`.
        Note that the values are referenced (not copied). Therefore, the
        modification of the returned ndarray will affet the data of the
        NNabla array.
        This method can be called as a setter to set the value held by this variable.

        Args:
            value(:obj:`numpy.ndarray`) (optional)

        Returns:
            :obj:`numpy.ndarray`
        """
        return self.data.data

    @d.setter
    def d(self, value):
        self.data.data[...] = value

    @property
    def g(self):
        """
        Returns the gradient values held by this variable, as a :class:`numpy.ndarray`.
        Note that the values are referenced (not copied). Therefore, the
        modification of the returned ndarray will affet the data of the
        NNabla array.
        This method can be called as a setter to set the gradient held by this variable.        

        Args:
            value(:obj:`numpy.ndarray`)

        Returns:
            :obj:`numpy.ndarray`
        """
        return self.grad.data

    @g.setter
    def g(self, value):
        self.grad.data[...] = value

    @property
    def parent(self):
        """
        Returns the parent function of this variable.
        This method can also be called as a setter.

        Args:
            func(:obj:`nnabla.function.Function`)

        Returns:
            :obj:`nnabla.function.Function`

        """
        cdef CgFunctionPtr cgf = self.varp.parent()
        if not cgf:
            return None
        return function.Function.create_from_c(cgf)

    @parent.setter
    def parent(self, func):
        cdef CgFunctionPtr cg_func = (< function.Function ?> func).fun
        assert cg_func, "TODO"
        self.varp.set_parent(cg_func)

    def forward(self, cpp_bool clear_buffer=False, cpp_bool clear_no_need_grad=False):
        """
        Performs a forward propagation from the root node to this variable.
        The forward propagation is performed on a subset of variables
        determined by the dependency of this variable.
        The subset is recursively constructed by tracking variables that the 
        variables in the subset depend on, starting from this variable,
        until it reaches the root variable(s) in the function graph.

        Args:
            clear_buffer (bool): Clear the no longer referenced variables
                during forward propagation to save memory.
                This is usually set as True in an inference
                or a validation phase. Default is False.
            clear_no_need_grad (bool): Clear the unreferenced variables with
                need_grad=False during forward propagation.
                True is usually used when calling this during training.
                This is ignored when clear_buffer=True.

        """
        with nogil:
            self.varp.forward(clear_buffer, clear_no_need_grad)

    def backward(self, grad=1, cpp_bool clear_buffer=False):
        """
        Performs a backward propagation starting from this variable until
        the root variable(s) is/are reached in the function graph.
        The propagation will stop at a variable with need_grad=False.

        Args:
            grad(scalar, :obj:`numpy.ndarray`, or :obj:`nnabla._nd_array.NdArray`):
                The gradient signal value(s) of this variable.
                The default value 1 is used in an usual neural network
                training. This option is useful if you have a gradient
                computation module outside NNabla, and want to use it as a
                gradient signal of the neural network built in NNabla.
                Note that this doesn't modifies the grad values of this
                variable.
            clear_buffer(bool): Clears the no longer referenced variables
                during backpropagation to save memory.  

        """
        cdef NdArrayPtr p
        if grad is None:
            pass
        elif np.isscalar(grad):
            arr = NdArray(self.shape)
            arr.fill(grad)
            p = ( < NdArray > arr).arr
        elif isinstance(grad, NdArray):
            p = ( < NdArray > grad).arr
        elif isinstance(grad, np.ndarray):
            arr = NdArray(grad.shape)
            arr.data = grad
            p = ( < NdArray > arr).arr
        else:
            # Try to interpret as scalar value
            arr = NdArray()
            arr.data = grad
            p = ( < NdArray > arr).arr

        with nogil:
            self.varp.backward(p, clear_buffer)

    def unlinked(self):
        """
        Gets unlinked (forgetting parent) variable that shares a Variable buffer
        instance.
        """
        var = Variable.create_from_cvariable(self.varp.variable().get().view())
        return var

    @property
    def persistent(self):
        """
        Returns the persistent flag of this variable. If True, the variable
        is not cleared even if clear options in
        :meth:`nnabla._variable.Variable.forward` and 
        :meth:`nnabla._variable.Variable.backward` are enabled.
        This is useful when you debug the variable values, or log them.
        This method can also be called as a setter.

        Args:
            b(bool)

        Returns: bool

        """
        return self.varp.persistent()

    @persistent.setter
    def persistent(self, cpp_bool b):
        self.varp.set_persistent(b)

    def visit(self, f):
        '''Visit functions recursively in forward order.

        Args:
            f (function): Function object which takes
                :obj:`nnabla._function.Function` object as an argument.

        Returns: None
        '''
        def _recursive_visit_functions(func, seen):
            if func is None:
                return
            seen.add(func)
            for i in func.inputs:
                if i.parent in seen:
                    continue
                _recursive_visit_functions(i.parent, seen)
            f(func)
        seen = set()
        _recursive_visit_functions(self.parent, seen)

    def visit_check(self, f):
        '''Visit functions recursively in forward order.

        Note:
            If any of evaluation of the function object returns True,
            the visit propagation will stop immediately,
            and will return True.

        Args:
            f (function): Function object which takes
                :obj:`nnabla._function.Function` object as an argument.

        Returns: bool
            Returns True if any of the function object call returns True.
        '''
        def _recursive_visit_functions(func, seen):
            if func is None:
                return False
            seen.add(func)
            for i in func.inputs:
                if i.parent in seen:
                    continue
                if _recursive_visit_functions(i.parent, seen):
                    return True
            return f(func)

        seen = set()
        return _recursive_visit_functions(self.parent, seen)

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
