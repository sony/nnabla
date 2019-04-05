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
