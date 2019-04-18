import nnabla.functions as F


class Distribution(object):
    def entropy(self):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def stddev(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def prob(self, x):
        raise NotImplementedError

    def sample(self, shape):
        raise NotImplementedError

    def sample_n(self, n):
        samples = [self.sample() for _ in range(n)]
        return F.stack(*samples, axis=1)
