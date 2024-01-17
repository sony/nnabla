import nnabla as nn
import numpy as np

from scipy import stats
from nnabla.experimental.distributions import Uniform
from distribution_test_util import distribution_test_util


def test_uniform():
    def param_fn(shape):
        low = np.random.random(shape)
        high = low + np.random.random(shape)
        return low, high

    def dist_fn(low, high):
        low = nn.Variable.from_numpy_array(low)
        high = nn.Variable.from_numpy_array(high)
        return Uniform(low=low, high=high)

    def scipy_fn(low, high):
        return stats.uniform(low, high - low)

    distribution_test_util(dist_fn, scipy_fn, param_fn)
