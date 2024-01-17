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


import numpy as np
import nnabla.functions as F

from .distribution import Distribution


class Uniform(Distribution):
    """Uniform distribution.

    Uniform distribution defined as :math:`x \sim U(low, high)`.
    Values are uniformly sampled between :math:`low` and :math:`high`.

    Args:
        low (~nnabla.Variable): N-D array of :math:`low` in definition.
        high (~nnabla.Variable): N-D arraya of :math:`high` in definition.

    """

    def __init__(self, low, high):
        assert low.shape == high.shape,\
            'For now, low and high must have same shape.'
        self.low = low
        self.high = high

    def mean(self):
        """Get mean of uniform distribution.

        .. math::

            \mu = low + \frac{high - low}{2}

        Returns:
            ~nnabla.Variable: N-D array.

        """
        return self.low + (self.high - self.low) / 2.0

    def stddev(self):
        """Get standard deviation of uniform distribution.

        .. math::

            \sigma = \frac{high - low}{\sqrt{12}}

        Returns:
            ~nnabla.Variable: N-D array.

        """
        return (self.high - self.low) / np.sqrt(12.0)

    def variance(self):
        """Get variance of uniform distribution.

        Returns:
            ~nnabla.Variable: N-D array defined as :math:`\sigma^2`.

        """
        return self.stddev() ** 2

    def prob(self, x):
        """Get probability of :math:`x` in uniform distribution.

        .. math::

            p(x | low, high) = \begin{cases}
                \frac{1}{high - low} & (x \geq low \text{and} x \leq high) \\
                0 & (otherwise)
            \end{cases}

        Args:
            x (~nnabla.Variable): N-D array.

        Returns:
            ~nnabla.Variable: N-D array.

        """
        return 1.0 / (self.high - self.low) * F.less(self.low, x) \
            * F.greater(self.high, x)

    def entropy(self):
        """Get entropy of uniform distribution.

        .. math::

            S = \ln(high - low)

        Returns:
            ~nnabla.Variable: N-D array.

        """
        return F.log(self.high - self.low)

    def sample(self, shape=None):
        """Sample points from uniform distribution.

        .. math::

            x \sim U(low, high)

        Args:
            shape (:obj:`tuple`): Shape of sampled points. If this is omitted,
                the returned shape is identical to
                :math:`high` and :math:`low`.

        Returns:
            ~nnabla.Variable: N-D array.

        """
        if shape is None:
            shape = self.high.shape
        eps = F.rand(low=0.0, high=1.0, shape=shape)
        return self.low + (self.high - self.low) * eps
