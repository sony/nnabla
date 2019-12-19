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


class MultivariateNormal(Distribution):
    """Multivariate normal distribution.

    Multivariate normal distribution defined as follows:

    .. math::

        p(x | \mu, \Sigma) = \frac{1}{\sqrt{(2 \pi)^k \det(\Sigma)}}
            \exp(-\frac{1}{2}(x - \mu)^T \Sigma^(-1) (x - \mu))

    where :math:`k` is a rank of `\Sigma`.

    Args:
        loc (~nnabla.Variable or numpy.ndarray): N-D array of :math:`\mu` in
            definition.
        scale (~nnabla.Variable or numpy.ndarray): N-D array of diagonal
            entries of :math:`L` such that covariance matrix
            :math:`\Sigma = L L^T`.

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
        """Get mean of multivariate normal distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array identical to :math:`\mu`.

        """
        # to avoid no parent error
        return F.identity(self.loc)

    def variance(self):
        """Get covariance matrix of multivariate normal distribution.

        .. math::

            \Sigma = L L^T

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        diag = self._diag_scale()
        return F.batch_matmul(diag, diag, False, True)

    def prob(self, x):
        """Get probability of `x` in multivariate normal distribution.

        .. math::

            p(x | \mu, \Sigma) = \frac{1}{\sqrt{(2 \pi)^k \det(\Sigma)}}
                \exp(-\frac{1}{2}(x - \mu)^T \Sigma^(-1) (x - \mu))

        Args:
            x (~nn.Variable): N-D array.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        k = self.loc.shape[1]
        z = 1.0 / ((2 * np.pi) ** k * F.batch_det(self._diag_scale())) ** 0.5

        diff = F.reshape(x - self.mean(), self.loc.shape + (1,), False)
        inv = F.batch_inv(self._diag_scale())
        y = F.batch_matmul(diff, inv, True, False)
        norm = F.reshape(F.batch_matmul(y, diff, False, False), (-1,), False)
        return z * F.exp(-0.5 * norm)

    def entropy(self):
        """Get entropy of multivariate normal distribution.

        .. math::

            S = \frac{1}{2} \ln \det(2 \pi e \Sigma)

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        det = F.batch_det(2.0 * np.pi * np.e * self._diag_scale())
        return 0.5 * F.log(det)

    def _diag_scale(self):
        return F.matrix_diag(self.scale)

    def sample(self, shape=None):
        """Sample points from multivariate normal distribution.

        .. math::

            x \sim N(\mu, \Sigma)

        Args:
            shape (:obj:`tuple`): Shape of sampled points. If this is omitted,
                the returned shape is identical to :math:`\mu`.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        if shape is None:
            shape = self.loc.shape
        eps = F.randn(mu=0.0, sigma=1.0, shape=shape)
        return self.mean() + self.scale * eps
