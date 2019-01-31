import nnabla as nn
import nnabla.parametric_functions as PF
import nnabla.functions as F
from nnabla.logger import logger
import numpy as np

def test_network(inp, output_stride, test=False):
    num_classes=21
    with nn.parameter_scope("1"):
        print(inp.shape)
        h = PF.convolution(inp, 32, (3,3), pad=(1,1), with_bias=False)
        h1=h.d
        h = F.relu(PF.batch_normalization(h, batch_stat=not test, eps=1e-03))
        #print(h.d)
    with nn.parameter_scope("2"):
        h = PF.convolution(h, 64, (3,3), pad=(1,1), with_bias=False)
        
        h = F.relu(PF.batch_normalization(h, batch_stat=not test, eps=1e-03))
        print(h.shape)

        h = PF.convolution(h, 21, (1,1), with_bias=False)
    return h

