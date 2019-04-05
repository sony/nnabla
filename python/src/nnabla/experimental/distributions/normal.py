import numpy as np
import nnabla.functions as F

from distribution import Distribution


class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def mean(self):
        # to avoid no parent error
        return self.loc + 0.0

    def stddev(self):
        # to avoid no parent error
        return self.scale + 0.0

    def variance(self):
        return self.stddev() ** 2

    def prob(self, x):
        z = 1.0 / (2 * np.pi * self.variance()) ** 0.5
        return  z * F.exp(-0.5 * ((x - self.mean()) ** 2) / self.variance())

    def entropy(self):
        return F.log(self.stddev()) + 0.5 * np.log(2.0 * np.pi * np.e)

    def sample(self, shape=None):
        if shape is None:
            shape = self.loc.shape
        eps = F.randn(mu=0.0, sigma=1.0, shape=shape)
        return self.mean() + self.stddev() * eps
