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


class Normal(Distribution):
    """Normal distribution.

    Normal distribution defined as follows:

    .. math::

        p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}
            \exp(-\frac{(x - \mu)^2}{2\sigma^2})

    Args:
        loc (~nnabla.Variable or numpy.ndarray): N-D array of :math:`\mu` in
            definition.
        scale (~nnabla.Variable or numpy.ndarray): N-D array of :math:`\sigma`
            in definition.

    """

    def __init__(self, loc, scale):
        assert loc.shape == scale.shape,\
            'For now, loc and scale must have same shape.'
        if isinstance(loc, np.ndarray):
            loc = nn.Variable.from_numpy_array(loc)
            loc.persistent = True
        if isinstance(scale, np.ndarray):
            scale = nn.Variable.from_numpy_array(scale)
            scale.persistent = True
        self.loc = loc
        self.scale = scale

    def mean(self):
        """Get mean of normal distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array identical to :math:`\mu`.

        """
        # to avoid no parent error
        return F.identity(self.loc)

    def stddev(self):
        """Get standard deviation of normal distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array identical to :math:`\sigma`.

        """
        # to avoid no parent error
        return F.identity(self.scale)

    def variance(self):
        """Get variance of normal distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array defined as :math:`\sigma^2`.

        """
        return self.stddev() ** 2

    def prob(self, x):
        """Get probability of :math:`x` in normal distribution.

        .. math::

            p(x | \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}}
                \exp(-\frac{(x - \mu)^2}{2\sigma^2})

        Args:
            x (~nnabla.Variable): N-D array.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        z = 1.0 / (2 * np.pi * self.variance()) ** 0.5
        return z * F.exp(-0.5 * ((x - self.mean()) ** 2) / self.variance())

    def entropy(self):
        """Get entropy of normal distribution.

        .. math::

            S = \frac{1}{2}\log(2 \pi e \sigma^2)

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        return F.log(self.stddev()) + 0.5 * np.log(2.0 * np.pi * np.e)

    def sample(self, shape=None):
        """Sample points from normal distribution.

        .. math::

            x \sim N(\mu, \sigma^2)

        Args:
            shape (:obj:`tuple`): Shape of sampled points. If this is omitted,
                the returned shape is identical to
                :math:`\mu` and :math:`\sigma`.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        if shape is None:
            shape = self.loc.shape
        eps = F.randn(mu=0.0, sigma=1.0, shape=shape)
        return self.mean() + self.stddev() * eps
