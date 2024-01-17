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


import nnabla.functions as F


class Distribution(object):
    """Distribution base class for distribution classes.
    """

    def entropy(self):
        """Get entropy of distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        raise NotImplementedError

    def mean(self):
        """Get mean of distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        raise NotImplementedError

    def stddev(self):
        """Get standard deviation of distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        raise NotImplementedError

    def variance(self):
        """Get variance of distribution.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        raise NotImplementedError

    def prob(self, x):
        """Get probability of sampled `x` from distribution.

        Args:
            x (~nnabla.Variable): N-D array.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        raise NotImplementedError

    def sample(self, shape):
        """Sample points from distribution.

        Args:
            shape (:obj:`tuple`): Shape of sampled points.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        raise NotImplementedError

    def sample_n(self, n):
        """Sample points from distribution :math:`n` times.

        Args:
            n (int): The number of sampling points.

        Returns:
            :class:`~nnabla.Variable`: N-D array.

        """
        samples = [self.sample() for _ in range(n)]
        return F.stack(*samples, axis=1)
