
import numpy as np

import nnabla.initializer as I


def w_init(x, out_dims, gain=0.02, type="xavier"):
    if type == "xavier":
        return I.NormalInitializer(sigma=I.calc_normal_std_glorot(x.shape[1], out_dims) * gain)

    raise ValueError("unsupported init type: {}.".format(type))


def pytorch_conv_init(inmaps, kernel):
    scale = 1 / np.sqrt(inmaps * np.prod(kernel))

    return I.UniformInitializer(lim=(-scale, scale))
