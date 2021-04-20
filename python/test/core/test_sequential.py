import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
import numpy as np
import pytest

from helper import create_temp_with_dir
from nnabla.testing import assert_allclose


class ConvLayer(nn.Module):
    def __init__(self, outmaps, kernel, stride=1, is_last=False):
        self.outmaps = outmaps
        self.kernel = (kernel, kernel)
        self.stride = (stride, stride)
        self.pad = (kernel // 2, kernel // 2)
        self.is_last = is_last

    def call(self, x):
        x = PF.convolution(x, outmaps=self.outmaps,
                           kernel=self.kernel, pad=self.pad, stride=self.stride)
        if self.is_last is False:
            x = F.relu(x)
        return x


class MaxPooling(nn.Module):
    def __init__(self, kernel, stride=1):
        self.kernel = (kernel, kernel)
        self.stride = (stride, stride)
        self.pad = (kernel // 2, kernel // 2)

    def call(self, x):
        y = F.max_pooling(x, kernel=self.kernel,
                          stride=self.stride, pad=self.pad)
        return y


class InceptionV1WithSequentialNOName(nn.Module):
    # Using nn.Sequential to construct network,
    # but don't apply a specify name for each layer within sequential container.

    def __init__(self, output_nc=1):
        self.conv1 = ConvLayer(16, kernel=3)
        self.conv2 = ConvLayer(32, kernel=3)

        # 1st branch
        self.branch1x1 = ConvLayer(64, kernel=1)

        # 2nd branch
        self.branch3x3 = nn.Sequential(
            ConvLayer(48, kernel=1),
            ConvLayer(64, kernel=3)
        )

        # 3rd branch
        self.branch5x5 = nn.Sequential(
            ConvLayer(64, kernel=1),
            ConvLayer(96, kernel=5)
        )

        # 4th branch
        self.branch_pool = nn.Sequential(
            MaxPooling(kernel=3),
            ConvLayer(32, kernel=1)
        )

    def call(self, x):
        h = self.conv1(x)
        y = self.conv2(h)
        f1 = self.branch1x1(y)
        f2 = self.branch3x3(y)
        f3 = self.branch5x5(y)
        f4 = self.branch_pool(y)

        out = F.concatenate(f1, f2, f3, f4, axis=1)
        return out


class InceptionV1WithSequentialWithName(nn.Module):
    # Using nn.Sequential to construct network,
    # at the same time, apply a specify name for each layer within sequential container.

    def __init__(self, output_nc=1):
        self.conv1 = ConvLayer(16, kernel=3)
        self.conv2 = ConvLayer(32, kernel=3)

        # 1st branch
        self.branch1x1 = ConvLayer(64, kernel=1)

        # 2nd branch
        self.branch3x3 = nn.Sequential(
            ('conv1', ConvLayer(48, kernel=1)),
            ('conv2', ConvLayer(64, kernel=3))
        )

        # 3rd branch
        self.branch5x5 = nn.Sequential(
            ('conv1', ConvLayer(64, kernel=1)),
            ('conv2', ConvLayer(96, kernel=5))
        )

        # 4th branch
        self.branch_pool = nn.Sequential(
            ('max_pooling', MaxPooling(kernel=3)),
            ('conv1', ConvLayer(32, kernel=1))
        )

    def call(self, x):
        h = self.conv1(x)
        y = self.conv2(h)
        f1 = self.branch1x1(y)
        f2 = self.branch3x3(y)
        f3 = self.branch5x5(y)
        f4 = self.branch_pool(y)

        out = F.concatenate(f1, f2, f3, f4, axis=1)
        return out


def DoubleConvUnit(i1, i2):
    double_conv_layer = nn.Sequential(
        ConvLayer(i1, kernel=3),
        ConvLayer(i2, kernel=3)
    )
    return double_conv_layer


class TSTSeqInSeqNetNoName(nn.Module):
    def __init__(self):
        self.conv1 = ConvLayer(8, kernel=3)
        self.conv_block2 = DoubleConvUnit(16, 32)
        self.conv_block3 = nn.Sequential(
            ConvLayer(64, kernel=3),
            DoubleConvUnit(16, 196)
        )

    def call(self, x):
        conv1 = self.conv1(x)
        conv_block2 = self.conv_block2(conv1)
        conv_block3 = self.conv_block3(conv_block2)

        return conv_block3


class TSTSeqInSeqNetWithName(nn.Module):
    def __init__(self):
        self.conv1 = ConvLayer(8, kernel=3)
        self.conv_block2 = DoubleConvUnit(16, 32)
        self.conv_block3 = nn.Sequential(
            ('single_conv', ConvLayer(64, kernel=3)),
            ('double_conv', DoubleConvUnit(16, 196))
        )

    def call(self, x):
        conv1 = self.conv1(x)
        conv_block2 = self.conv_block2(conv1)
        conv_block3 = self.conv_block3(conv_block2)

        return conv_block3


nnp_file = "t.nnp"


@pytest.mark.parametrize("TSTNet", [InceptionV1WithSequentialNOName, InceptionV1WithSequentialWithName, TSTSeqInSeqNetNoName, TSTSeqInSeqNetWithName])
def test_sequential(TSTNet):
    tst_net = TSTNet()

    input_shape = (2, 3, 16, 16)
    input = nn.Variable.from_numpy_array((np.random.random(input_shape)))

    y = tst_net(input)
    y.forward()

    x = nn.ProtoVariable(input_shape)
    with nn.graph_def.graph() as g1:
        h = tst_net(x)

    with create_temp_with_dir(nnp_file) as tmp_file:
        g1.save(tmp_file)

        g2 = nn.graph_def.load(tmp_file)
        for net in g2.networks.values():
            out = net(input)
            out.forward()
        # should equal
        assert_allclose(out.d, y.d)
