import nnabla as nn
import numpy as np

from scipy import stats
from nnabla.experimental.distributions import Normal
from distribution_test_util import distribution_test_util


def test_normal():
    def param_fn(shape):
        loc = np.random.random(shape)
        scale = np.random.random(shape) + 1e-5
        return loc, scale

    def dist_fn(loc, scale):
        loc = nn.Variable.from_numpy_array(loc)
        scale = nn.Variable.from_numpy_array(scale)
        return Normal(loc=loc, scale=scale)

    def scipy_fn(loc, scale):
        return stats.norm(loc, scale)

    distribution_test_util(dist_fn, scipy_fn, param_fn)
