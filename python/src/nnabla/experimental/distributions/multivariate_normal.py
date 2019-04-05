import numpy as np
import nnabla.functions as F

from .distribution import Distribution


class MultivariateNormal(Distribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def mean(self):
        # to avoid no parent error
        return self.loc + 0.0

    def variance(self):
        diag = self._diag_scale()
        return F.batch_matmul(diag, diag, False, True)

    def prob(self, x):
        k = self.loc.shape[1]
        z = 1.0 / ((2 * np.pi) ** k * F.batch_det(self._diag_scale())) ** 0.5

        diff = F.reshape(x - self.mean(), self.loc.shape + (1,), False)
        inv = F.batch_inv(self._diag_scale())
        y = F.batch_matmul(diff, inv, True, False)
        norm = F.reshape(F.batch_matmul(y, diff, False, False), (-1,), False)
        return z * F.exp(-0.5 * norm)

    def entropy(self):
        det = F.batch_det(2.0 * np.pi * np.e * self._diag_scale())
        return 0.5 * F.log(det)

    def _diag_scale(self):
        return F.matrix_diag(self.scale)

    def sample(self, shape=None):
        if shape is None:
            shape = self.loc.shape
        eps = F.randn(mu=0.0, sigma=1.0, shape=shape)
        return self.mean() + self.scale * eps
