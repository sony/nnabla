import numpy as np
import nnabla.functions as F

from .distribution import Distribution


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def mean(self):
        return self.low + (self.high - self.low) / 2.0

    def stddev(self):
        return (self.high - self.low) / np.sqrt(12.0)

    def variance(self):
        return self.stddev() ** 2

    def prob(self, x):
        return 1.0 / (self.high - self.low)

    def entropy(self):
        return F.log(self.high - self.low)

    def sample(self, shape=None):
        if shape is None:
            shape = self.high.shape
        eps = F.rand(low=0.0, high=1.0, shape=shape)
        return self.low + (self.high - self.low) * eps
