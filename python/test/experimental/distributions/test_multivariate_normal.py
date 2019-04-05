import numpy as np
import nnabla as nn

from scipy import stats
from nnabla.experimental.distributions import MultivariateNormal
from distribution_test_util import distribution_test_util


def test_multivariate_normal():
    def param_fn(shape):
        loc = np.random.random(shape)
        scale = np.random.random(shape) + 1e-5
        return loc, scale

    def dist_fn(loc, scale):
        loc = nn.Variable.from_numpy_array(loc)
        scale = nn.Variable.from_numpy_array(scale)
        return MultivariateNormal(loc=loc, scale=scale)

    def scipy_fn(loc, scale):
        return stats.multivariate_normal(np.reshape(loc, (-1,)),
                                         np.reshape(scale, (-1,)))

    def ref_sample_fn(loc, scale, shape):
        return np.random.normal(loc, scale, size=shape)

    def ref_entropy(loc, scale, shape):
        entropy = np.zeros(shape[0])
        for i, (l, s) in enumerate(zip(loc, scale)):
            entropy[i] = stats.multivariate_normal.entropy(l, s)
        return entropy

    def ref_prob_fn(loc, scale, sample, shape):
        probs = np.zeros(shape[0])
        for i, (l, s) in enumerate(zip(loc, scale)):
            probs[i] = stats.multivariate_normal.pdf(sample[i], l, s)
        return probs

    # due to memory error at scipy with large dimension, use small sample size
    distribution_test_util(dist_fn, scipy_fn, param_fn,
                           skip_columns=['mean', 'stddev', 'variance'],
                           sample_shape=(100, 10), ref_sample_fn=ref_sample_fn,
                           ref_columns={'entropy': ref_entropy},
                           ref_prob_fn=ref_prob_fn)
