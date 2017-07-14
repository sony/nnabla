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

from __future__ import absolute_import

from ._variable import Context
from ._variable import Variable as _Variable


# As Cython does not support some arithmetic operations,
# Variable is inherited in order to define arithmetic operations.
class Variable(_Variable):
    __doc__ = _Variable.__doc__

    def __repr__(self):
        return "<Variable({}, need_grad={}) at {}>".format(
            self.shape, self.need_grad, hex(id(self)))

    def __add__(self, other):
        """
        Element-wise addition.
        Implements the addition operator expression ``A + B``, together with :func:`~nnabla.variable.__radd__` .
        When a scalar is specified for ``other``, this function performs an
        element-wise operation for all elements in ``self``.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.add2` or
                :func:`~nnabla.functions.add_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.add2(self, other)
        return F.add_scalar(self, other)

    def __mul__(self, other):
        """
        Element-wise multiplication.
        Implements the multiplication operator expression ``A * B``, together with :func:`~nnabla.variable.__rmul__` .
        When a scalar is specified for ``other``, this function performs an
        element-wise operation for all elements in ``self``.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.mul2` or
                :func:`~nnabla.functions.mul_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.mul2(self, other)
        return F.mul_scalar(self, other)

    def __sub__(self, other):
        """
        Element-wise subtraction.
        Implements the subtraction operator expression ``A - B``, together with :func:`~nnabla.variable.__rsub__` .
        When a scalar is specified for ``other``, this function performs an
        element-wise operation for all elements in ``self``.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.sub2` or
                :func:`~nnabla.functions.add_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.sub2(self, other)
        return F.add_scalar(self, -other)

    def __rsub__(self, other):
        """
        Element-wise subtraction.
        Part of the implementation of the subtraction operator.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.sub2` or
                :func:`~nnabla.functions.r_sub_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.sub2(other, self)
        return F.r_sub_scalar(self, other)

    def __div__(self, other):
        """
        Element-wise division.
        Implements the division operator expression ``A / B``, together with :func:`~nnabla.variable.__rdiv__` .
        When a scalar is specified for ``other``, this function performs an
        element-wise operation for all elements in ``self``.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.div2` or
                :func:`~nnabla.functions.mul_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.div2(self, other)
        return F.mul_scalar(self, 1. / other)

    def __rdiv__(self, other):
        """
        Element-wise division.
        Part of the implementation of the division operator.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.sub2` or
                :func:`~nnabla.functions.add_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.div2(other, self)
        return F.r_div_scalar(self, other)

    def __pow__(self, other):
        """
        Element-wise power function.
        Implements the power operator expression ``A ** B``, together with :func:`~nnabla.variable.__rpow__` .
        When a scalar is specified for ``other``, this function performs an
        element-wise operation for all elements in ``self``.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.pow2` or
                :func:`~nnabla.functions.pow_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.pow2(self, other)
        return F.pow_scalar(self, other)

    def __rpow__(self, other):
        """
        Element-wise power function.
        Part of the implementation of the power operator expression.

        Args:
            other (float or ~nnabla.Variable): Internally calling
                :func:`~nnabla.functions.pow2` or
                :func:`~nnabla.functions.r_pow_scalar` according to the
                type.

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        if isinstance(other, Variable):
            return F.pow2(other, self)
        return F.r_pow_scalar(self, other)

    def __radd__(self, other):
        """
        Element-wise addition.
        Part of the implementation of the addition operator expression.

        Args:
            other (float or ~nnabla.Variable): Replace with :meth:`__add__`.

        Returns: :class:`nnabla.Variable`

        """
        return self + other

    def __rmul__(self, other):
        """
        Element-wise division.
        Part of the implementation of the division operator expression.

        Args:
            other (float or ~nnabla.Variable): Replace with :meth:`__mul__`.

        Returns: :class:`nnabla.Variable`

        """
        return self * other

    def __truediv__(self, other):
        """
        Element-wise division.
        Part of the implementation of the division operator expression.
        """
        return self.__div__(other)

    def __rtruediv__(self, other):
        """
        Element-wise division.
        Part of the implementation of the division operator expression.
        """
        return self.__rdiv__(other)

    def __pos__(self):
        """
        This function simply returns itself.
        Implements the unary plus operator, ``+A``.

        Returns: :class:`nnabla.Variable`

        """
        return self

    def __neg__(self):
        """
        Element-wise negation.
        Implements the unary negation expression ``-A`` .

        Returns: :class:`nnabla.Variable`

        """
        import nnabla.functions as F
        return F.mul_scalar(self, -1)
