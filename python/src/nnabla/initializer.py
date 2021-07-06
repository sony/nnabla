# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
from . import random

# Use it like "random_float_type(x)", not ".astype(random_float_type)"
# because this manner is applicable to both numpy.array and 0-dimensional
# numpy.array (or Python scalar) which appears when Initializer takes shape=(),
# for example self.rng.randn(*shape) where shape = ().
random_float_type = np.float32


class BaseInitializer(object):

    """Base class of the parameter initializer.

    """

    def __call__(self, shape):
        """Generates an array with an initializer.

        Args:
            shape (:obj:`tuple` of :obj:`int`): :obj:`numpy.ndarray` with the shape created.

        Returns:
            :obj:`numpy.ndarray` : Array.

        Note:
            Subclasses of :class:`~nnabla.initializer.BaseInitializer` must override this method.

        """
        raise NotImplementedError()


class NormalInitializer(BaseInitializer):

    r"""Generates a random array from a specified normal distribution.

    .. math::
        \mathbf x \sim {\cal N} (\mathbf 0 | \sigma^2 \mathbf I)

    Args:
        sigma (float): :math:`\sigma`.
        rng (numpy.random.RandomState): Random number generator.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        w = I.NormalInitializer(5e-5)
        b = I.NormalInitializer(0.0)
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')
    """

    def __init__(self, sigma=1.0, rng=None):
        if rng is None:
            rng = random.prng
        self.rng = rng
        self.sigma = sigma

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.sigma)

    def __call__(self, shape):
        return random_float_type(self.rng.randn(*shape) * self.sigma)


class UniformInitializer(BaseInitializer):

    r"""Generates a random array from a specified uniform distribution.

    .. math::
        \mathbf x \sim {\cal U} (a, b)

    Args:
        lim (:obj:`tuple` of :obj:`float`): A tuple of two floats, :math:`(a, b)`.
        rng (numpy.random.RandomState): Random number generator.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        w = I.UniformInitializer() # this generates uniform distribution within the default range of (-1,1)
        b = I.UniformInitializer((-0.5,0.5))
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')
    """

    def __init__(self, lim=(-1, 1), rng=None):
        if rng is None:
            rng = random.prng
        self.rng = rng
        self.lim = lim

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               repr(self.lim))

    def __call__(self, shape):
        return random_float_type(self.rng.uniform(self.lim[0], self.lim[1],
                                                  size=shape))


class UniformIntInitializer(BaseInitializer):

    r"""Generates a random array from a specified integer uniform distribution.

    .. math::
        \mathbf x \sim {\cal U} ([a, b))

    Args:
        lim (:obj:`tuple` of :obj:`int`): A tuple of two ints, :math:`(a, b)`.
        rng (numpy.random.RandomState): Random number generator.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        w = I.UniformIntInitializer() # this generates uniform integer distribution within the default range of (0,10)
        b = I.UniformIntInitializer((-1,1))
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')
    """

    def __init__(self, lim=(0, 10), rng=None):
        if rng is None:
            rng = random.prng
        self.rng = rng
        self.lim = lim

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               repr(self.lim))

    def __call__(self, shape):
        return self.rng.randint(self.lim[0], self.lim[1], size=shape)


class RangeInitializer(BaseInitializer):

    """Generates an array with sequence of numbers.

    .. math::
        \mathbf x[i] = start + step * i

    Args:
        start (int): A start value.
        step (int): A step value.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.initializer as I

        x = nn.Variable([100])
        x.d = I.RangeInitializer(0, 1)(x.shape)
    """

    def __init__(self, start=0, step=1):
        self.start = start
        self.step = step

    def __call__(self, shape):
        a = np.arange(0, shape[-1], 1)
        return np.broadcast_to(self.start + a * self.step, shape)


class ConstantInitializer(BaseInitializer):

    """Generates a constant valued array.

    Args:
        value (float): A constant value.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        w = I.ConstantInitializer(0.1)
        b = I.ConstantInitializer() # this generates constant valued array of default value 0
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv'
    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape):
        return random_float_type(np.ones(shape) * self.value)


class OrthogonalInitializer(BaseInitializer):

    r"""Generates an orthogonal matrix weights proposed by Saxe et al.

    Args:
        gain (float): scaling factor which should be decided depending on a type of units.
        rng (numpy.random.RandomState): Random number generator.

    Example:

    .. code-block:: python

        import numpy as np
        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        w = I.OrthogonalInitializer(np.sqrt(2.0))
        b = I.ConstantInitializer(0.0)
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')

    References:
        * `Saxe, et al. Exact solutions to the nonlinear dynamics of
          learning in deep linear neural networks.
          <https://arxiv.org/abs/1312.6120>`_
    """

    def __init__(self, gain=1.0, rng=None):
        if rng is None:
            rng = random.prng
        self.rng = rng
        self.gain = gain

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.gain)

    def __call__(self, shape):
        flat_shape = (shape[0], int(np.prod(shape[1:])))
        x = self.rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(x, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return random_float_type(q.reshape(shape) * self.gain)


class WeightNormalizationScaleInitializer(BaseInitializer):

    r"""Compute the L2-norm for each weight kernel.

    This initializer is specific to the weight normalization scale to keep the same magnitude of the originally initialized weights even after the applicaiton of the weight normalization at only initialization.

    Args:
        w (:obj:`Variable`): Weight the weight normalization is applied.
        dim (:obj:`int`): Output dimension of the weight normalization.
        eps (:obj:`float`): Eplision of the weight normalization.
    """

    def __init__(self, w, dim=0, eps=1e-12):
        self.w = w.get_unlinked_variable()
        self.dim = dim
        self.eps = eps

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__)

    def __call__(self, shape):
        axis = tuple([a for a in range(len(self.w.shape)) if a != self.dim])
        w_norm_data = np.sqrt(np.sum(self.w.d ** 2, axis=axis) + self.eps)
        return random_float_type(w_norm_data)


def calc_normal_std_he_forward(inmaps, outmaps, kernel=(1, 1)):
    r"""Calculates the standard deviation proposed by He et al.

    .. math::
        \sigma = \sqrt{\frac{2}{NK}}

    Args:
        inmaps (int): Map size of an input Variable, :math:`N`.
        outmaps (int): Map size of an output Variable, :math:`M`.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel spatial shape.
            In above definition, :math:`K` is the product of shape dimensions.
            In Affine, the default value should be used.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        s = I.calc_normal_std_he_forward(x.shape[1],64)
        w = I.NormalInitializer(s)
        b = I.ConstantInitializer(0)
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')

    References:
        * `He, et al. Delving Deep into Rectifiers: Surpassing Human-Level
          Performance on ImageNet Classification.
          <https://arxiv.org/abs/1502.01852>`_

    """
    return np.sqrt(2. / (np.prod(kernel) * inmaps))


def calc_normal_std_he_backward(inmaps, outmaps, kernel=(1, 1)):
    r"""Calculates the standard deviation of He et al. (backward case).

    .. math::
        \sigma = \sqrt{\frac{2}{MK}}

    Args:
        inmaps (int): Map size of an input Variable, :math:`N`.
        outmaps (int): Map size of an output Variable, :math:`M`.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel spatial shape.
            In above definition, :math:`K` is the product of shape dimensions.
            In Affine, the default value should be used.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        s = I.calc_normal_std_he_backward(x.shape[1],64)
        w = I.NormalInitializer(s)
        b = I.ConstantInitializer(0)
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')

    References:
        * `He, et al. Delving Deep into Rectifiers: Surpassing Human-Level
          Performance on ImageNet Classification.
          <https://arxiv.org/abs/1502.01852>`_

    """
    return np.sqrt(2. / (np.prod(kernel) * outmaps))


def calc_normal_std_glorot(inmaps, outmaps, kernel=(1, 1)):
    r"""Calculates the standard deviation proposed by Glorot et al.

    Note: 
        We have updated the definition as following from v.1.2. It may affect the
        behavior of existing scripts that rely on the default initialization.

    .. math::
        \sigma = \sqrt{\frac{2}{K(N + M)}}

    Args:
        inmaps (int): Map size of an input Variable, :math:`N`.
        outmaps (int): Map size of an output Variable, :math:`M`.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel spatial shape.
            In above definition, :math:`K` is the product of shape dimensions.
            In Affine, the default value should be used.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        s = I.calc_normal_std_glorot(x.shape[1],64)
        w = I.NormalInitializer(s)
        b = I.ConstantInitializer(0)
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')

    References:
        * `Glorot and Bengio. Understanding the difficulty of training deep
          feedforward neural networks
          <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`_

    """
    return np.sqrt(2. / (np.prod(kernel) * (inmaps + outmaps)))


def calc_uniform_lim_glorot(inmaps, outmaps, kernel=(1, 1)):
    r"""Calculates the lower bound and the upper bound of the uniform distribution proposed by Glorot et al.

    Note: 
        We have updated the definition as following from v.1.3. It may affect the
        behavior of existing scripts that rely on the default initialization.

    .. math::
        b &= \sqrt{\frac{6}{K(N + M)}}\\
        a &= -b

    Args:
        inmaps (int): Map size of an input Variable, :math:`N`.
        outmaps (int): Map size of an output Variable, :math:`M`.
        kernel (:obj:`tuple` of :obj:`int`): Convolution kernel spatial shape.
            In above definition, :math:`K` is the product of shape dimensions.
            In Affine, the default value should be used.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.initializer as I

        x = nn.Variable([60,1,28,28])
        lb,ub= I.calc_uniform_lim_glorot(x.shape[1],64)
        w = I.UniformInitializer((lb,ub))
        b = I.ConstantInitializer(0)
        h = PF.convolution(x, 64, [3, 3], w_init=w, b_init=b, pad=[1, 1], name='conv')

    References:
        * `Glorot and Bengio. Understanding the difficulty of training deep
          feedforward neural networks
          <http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf>`_

    """

    d = np.sqrt(6. / (np.prod(kernel) * (inmaps + outmaps)))
    return -d, d
