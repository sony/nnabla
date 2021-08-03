from _variable cimport Variable
from _nd_array cimport NdArray

cdef object pos(object self):
    """
    This function simply returns itself.
    Implements the unary plus operator, ``+A``.

    Returns: :class:`nnabla.Variable` or :class:`nnabla.NdArray`

    """
    return self

cdef object neg(object self):
    """
    Element-wise negation.
    Implements the unary negation expression ``-A`` .

    Returns: :class:`nnabla.Variable` or :class:`nnabla.NdArray`

    """
    import nnabla.functions as F
    return F.mul_scalar(self, -1)

cdef object add(object x, object y):
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

cdef object sub(object x, object y):
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

cdef object mul(object x, object y):
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

cdef object truediv(object x, object y):
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
            return F.mul_scalar(x, 1. / y)
    else:
        if isinstance(y, (NdArray, Variable)):
            return F.r_div_scalar(y, x)
        else:
            return x / y

cdef object div(object x, object y):
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
            return F.mul_scalar(x, 1. / y)
    else:
        if isinstance(y, (NdArray, Variable)):
            return F.r_div_scalar(y, x)
        else:
            return x / y

cdef object pow(object x, object y, object z):
    """
    Element-wise power function.

    Implements the power operator expression ``x ** y``,
    optionally ``x ** y % z`` (but not implemented).
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


cdef object matmul(object x, object y):
    """
    Matrix multiplication

    Implements the matmul operator expression ``x @ y``.
    When both of ``x`` and ``y`` are either :obj:`~nnabla.Variable` or
    :obj:`~nnabla.NdArray`, :func:`~nnabla.functions.affine`, :func:`~nnabla.functions.sum`,
    :func:`~nnabla.functions.reshape`, :func:`~nnabla.functions.batch_matmul` are internally called.

    Note:
        If both arguments are 1-D, it is inner product of vectors.
        If both arguments are 2-D, it is matrix multiplication.
        If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to the first argument's dimensions.
        After matrix multiplication the prepended 1 is removed.
        If the second argument is 1-D, is it promoted to a matrix by appending a 1 to the second argument's dimensions.
        After matrix multiplication the appended 1 is removed.
        If either arguments is N-D, N>2, it is treated as the batch matrix multiplication and broadcast accordingly.
        e.g. If the first argument x is (p, j, n, k) and the second argument y is (k, m), the out is (p, j, n, m).

    Args:
        x (~nnabla.Variable or ~nnabla.NdArray): Left operand. Input array, scalar not allowed.
        y (~nnabla.Variable or ~nnabla.NdArray): Right operand. Input array, scalar not allowed.

    Returns: :class:`~nnabla.Variable` or :class:`~nnabla.NdArray`.

    Examples:

    .. code-block:: python

        import numpy as np
        import nnabla as nn

        # vector * vector
        arr1 = np.random.random([10]).astype(np.float32)
        arr2 = np.random.random([10]).astype(np.float32)
        n1 = nn.NdArray.from_numpy_array(arr1)
        n2 = nn.NdArray.from_numpy_array(arr2)
        v1 = nn.Variable.from_numpy_array(arr1)
        v2 = nn.Variable.from_numpy_array(arr2)
        ans1 = n1 @ n2
        ans2 = v1 @ v2
        ans2.forward()
        print(ans1.shape)
        print(ans2.shape)
        # ()
        # ()

        # matrix * vector
        arr1 = np.random.random([10, 5]).astype(np.float32)
        arr2 = np.random.random([5]).astype(np.float32)
        n1 = nn.NdArray.from_numpy_array(arr1)
        n2 = nn.NdArray.from_numpy_array(arr2)
        v1 = nn.Variable.from_numpy_array(arr1)
        v2 = nn.Variable.from_numpy_array(arr2)
        ans1 = n1 @ n2
        ans2 = v1 @ v2
        ans2.forward()
        print(ans1.shape)
        print(ans2.shape)
        # (10, )
        # (10, )

        # matrix * broadcasted vector
        arr1 = np.random.random([10, 5, 2]).astype(np.float32)
        arr2 = np.random.random([2]).astype(np.float32)
        n1 = nn.NdArray.from_numpy_array(arr1)
        n2 = nn.NdArray.from_numpy_array(arr2)
        v1 = nn.Variable.from_numpy_array(arr1)
        v2 = nn.Variable.from_numpy_array(arr2)
        ans1 = n1 @ n2
        ans2 = v1 @ v2
        ans2.forward()
        print(ans1.shape)
        print(ans2.shape)
        # (10, 5)
        # (10, 5)

    """
    import nnabla.functions as F
    assert isinstance(x, (NdArray, Variable)) and isinstance(y, (NdArray, Variable)), "All inputs must be ~nnabla.Variable or ~nnabla.NdArray"
    assert x.ndim != 0, "1st input operand has 0 dimension, does not have enough dimensions (func core with signature (m?,k),(k,n?)->(m?,n?) requires 1)"
    assert y.ndim != 0, "2nd input operand has 0 dimension, does not have enough dimensions (func core with signature (m?,k),(k,n?)->(m?,n?) requires 1)"

    if x.ndim == 1 and y.ndim == 1:
        result = F.sum(x * y)
    elif x.ndim == 2 and y.ndim==2:
        result = F.affine(x, y)
    elif x.ndim == 1 and y.ndim == 2:
        h = F.affine(F.reshape(x, (1, -1)), y)
        result = F.reshape(h, (h.shape[-1],))
    elif x.ndim == 2 and y.ndim == 1:
        h = F.affine(x, F.reshape(y, (-1, 1)))
        result = F.reshape(h, h.shape[:-1])
    elif x.ndim > 2 or y.ndim > 2:
        if x.ndim == y.ndim:
            result = F.batch_matmul(x, y)
        else:
            data_length = x.ndim - y.ndim
            if data_length > 0:
                shape = list(y.shape)
                if y.ndim == 1:
                    shape.insert(1, 1)
                    data_length -= 1
                for i in range(0, data_length):
                    shape.insert(0, 1)
                y_ = F.reshape(y, tuple(shape))
                result = F.batch_matmul(x, y_)
                if y.ndim == 1:
                    result = F.reshape(result, result.shape[:-1])
            else:
                data_length = -data_length
                shape = list(x.shape)
                for i in range(0, data_length):
                    shape.insert(0, 1)
                x_ = F.reshape(x, tuple(shape))
                result = F.batch_matmul(x_, y)
                if x.ndim == 1:
                    shape_ = list(result.shape)
                    shape_.pop(result.ndim - 2)
                    result = F.reshape(result, shape_)

    return result
