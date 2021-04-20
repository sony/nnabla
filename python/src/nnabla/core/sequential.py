import nnabla as nn


class Sequential(nn.Module):
    """A sequential block.
    User may construct their network by a sequential block. Importantly, the component within sequential block must be
    an instance of nn.Module.

    For intuitive understanding, some small examples as follows:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.functions as F

        class ConvLayer(nn.Module):
            def __init__(self, outmaps, kernel, stride=1, pad=0):
                self.outmaps = outmaps
                self.kernel = (kernel, kernel)
                self.pad = (pad, pad)
                self.stride = (stride, stride)

            def call(self, x):
                x = PF.convolution(x, outmaps=self.outmaps, kernel=self.kernel, pad=self.pad, stride=self.stride)
                x = F.relu(x)
                return x

        # Example of using Sequentional
        layer = nn.Sequential(
            ConvLayer(48, kernel=1),
            ConvLayer(64, kernel=3, pad=1)
        )

        # Example of using Sequentional with a specify name for each layer
        layer = nn.Sequential(
            ('conv1', ConvLayer(48, kernel=1)),
            ('conv2', ConvLayer(64, kernel=3, pad=1))
        )
    """

    def __init__(self, *args, **kwargs):
        for idx, module in enumerate(args):
            if not isinstance(module, tuple):
                if isinstance(module, nn.Module):
                    self.submodules[module.name + '_' + str(idx)] = module
                else:
                    raise TypeError(
                        "The input layer {}/{} should be a instance of nn.Module".format(self.name,
                                                                                         module.__class__.__name__))
            else:
                if isinstance(module[1], nn.Module):
                    self.submodules[module[0]] = module[1]
                else:
                    raise TypeError(
                        "The input layer {}/{} should be a instance of nn.Module".format(self.name,
                                                                                         module[1].__class__.__name__))

    def call(self, *args, **kwargs):
        for module in self.submodules.values():
            args = module(*args, **kwargs)
            result = args
            if args is not None:
                if not isinstance(args, tuple):
                    args = (args,)

        return result
