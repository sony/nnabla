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
