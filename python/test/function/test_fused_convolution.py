# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nbla_test_utils import list_context
from nnabla.testing import assert_allclose

ctxs = list_context('FusedConvolution')
cpu_context = nn.Context(["cpu:float"])


class RefFusedConvolutionGraph(object):

    def __init__(self, x, weight, bias, beta, gamma, rmean, rvar, z,
                 base_axis, pad, stride, dilation, group, channel_last,
                 decay_rate, eps, batch_stat,
                 nonlinearity, nonlinearity_args):

        from collections import OrderedDict
        inputs = OrderedDict()
        xvar = nn.Variable.from_numpy_array(x)
        weightvar = nn.Variable.from_numpy_array(weight)
        inputs['x'] = xvar
        inputs['weight'] = weightvar
        biasvar = None
        betavar = None
        gammavar = None
        rmeanvar = None
        rvarvar = None
        zvar = None
        if bias is not None:
            biasvar = nn.Variable.from_numpy_array(bias)
            inputs['bias'] = biasvar
        if beta is not None:
            betavar = nn.Variable.from_numpy_array(beta)
            gammavar = nn.Variable.from_numpy_array(gamma)
            rmeanvar = nn.Variable.from_numpy_array(rmean)
            rvarvar = nn.Variable.from_numpy_array(rvar)
            inputs['beta'] = betavar
            inputs['gamma'] = gammavar
            inputs['rmean'] = rmeanvar
            inputs['rvar'] = rvarvar
        if z is not None:
            zvar = nn.Variable.from_numpy_array(z)
            inputs['z'] = zvar
        h = F.convolution(xvar, weightvar, biasvar, base_axis,
                          pad, stride, dilation, group, channel_last)
        if beta is not None:
            h = F.batch_normalization(h, betavar, gammavar, rmeanvar, rvarvar,
                                      [h.ndim - 1 if channel_last else base_axis],
                                      decay_rate, eps, batch_stat)
        if z is not None:
            h = F.add2(h, zvar)
        h = ref_activation(h, nonlinearity, nonlinearity_args)
        self.input_dict = inputs
        self.output = h

    def get_output(self):
        self.output.forward(clear_buffer=True)
        return self.output.data.get_data('r')

    def get_grads(self, grad, need_grad_flags):
        ng_flags = [ng for ng in need_grad_flags if ng is not None]
        for ivar, ng in zip(self.input_dict.values(), ng_flags):
            ivar.need_grad = ng
            ivar.grad.zero()
        self.output.forward(clear_no_need_grad=True)
        self.output.backward(grad, clear_buffer=True)
        return np.concatenate([ivar.grad.get_data('r').flatten() for (ivar, ng) in zip(self.input_dict.values(), ng_flags) if ng])


def ref_activation(x, nonlinearity, nonlinearity_args):
    if nonlinearity == 'identity' or not nonlinearity:
        return x
    elif nonlinearity == 'relu':
        return F.relu(x)
    elif nonlinearity == 'sigmoid':
        return F.sigmoid(x)
    elif nonlinearity == 'tanh':
        return F.tanh(x)
    elif nonlinearity == 'leaky_relu':
        return F.leaky_relu(x, nonlinearity_args[0])
    elif nonlinearity == 'elu':
        return F.elu(x, nonlinearity_args[0])
    elif nonlinearity == 'relu6':
        return F.relu6(x)
    raise ValueError("unknown nonlinearity type {}".format(nonlinearity))


def ref_fused_convolution(x, weight, bias, beta, gamma, rmean, rvar, z,
                          base_axis, pad, stride, dilation, group, channel_last,
                          decay_rate, eps, batch_stat,
                          nonlinearity, nonlinearity_args):

    with nn.context_scope(cpu_context):
        graph = RefFusedConvolutionGraph(**locals())
    return graph.get_output()


def ref_grad_fused_convolution(x, weight, bias, beta, gamma, rmean, rvar, z, dy,
                               base_axis, pad, stride, dilation, group, channel_last,
                               decay_rate, eps, batch_stat,
                               nonlinearity, nonlinearity_args, need_grad_flags):
    args = locals().copy()
    del args['dy']
    del args['need_grad_flags']
    with nn.context_scope(cpu_context):
        graph = RefFusedConvolutionGraph(**args)
    return graph.get_grads(dy, need_grad_flags=need_grad_flags)


def create_inputs(rng, kernel, pad, stride, dilation, group, with_bias, bn, add2, channel_last):
    from refs import get_conv_out_size
    bs = 2
    hw = (5, 5)
    ohw = tuple(get_conv_out_size(s, kernel, pad, stride, dilation)
                for s in hw)
    ichannels = 4
    ochannels = 6
    kernels = (kernel, kernel)
    inshape = (bs, ichannels) + hw
    outshape = (bs, ochannels) + ohw
    wshape = (ochannels, ichannels // group,) + kernels

    def to_channel_last(shape):
        return tuple(shape[i] for i in (0, 2, 3, 1))
    if channel_last:
        inshape = to_channel_last(inshape)
        outshape = to_channel_last(outshape)
        wshape = to_channel_last(wshape)
    x = rng.randn(*inshape).astype(np.float32)
    weight = rng.randn(*wshape).astype(np.float32)
    bias = None
    if with_bias:
        bias = rng.randn(ochannels).astype(np.float32)
    if bn is not None:
        axis = 3 if channel_last else 1
        shape_stat = [1 for _ in range(x.ndim)]
        shape_stat[axis] = ochannels
        beta = rng.randn(*shape_stat).astype(np.float32)
        gamma = rng.randn(*shape_stat).astype(np.float32)
        rmean = np.zeros(shape_stat, dtype=np.float32)
        rvar = np.ones(shape_stat, dtype=np.float32)
    else:
        beta = None
        gamma = None
        rmean = None
        rvar = None
    if add2:
        # Note: The last dimension must be a multiple of 4
        # if we want to test cudnn BN persistent mode.
        z = rng.randn(*outshape).astype(np.float32)
    else:
        z = None

    return x, weight, bias, beta, gamma, rmean, rvar, z


@pytest.mark.parametrize("seed", [1223, 726])
@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("kernel", [3])
@pytest.mark.parametrize("pad", [1])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("group", [2])
@pytest.mark.parametrize("decay_rate", [0.9])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("with_bias, bn, add2, channel_last", [
    (False, True, True, False),
    (False, True, False, False),
    # TODO will make channel_last=True once conv cpu supports it
    (True, None, True, False),
    (False, False, False, False),
    (True, None, False, False),
])
@pytest.mark.parametrize("nonlinearity, nonlinearity_args", [
    ('identity', []),
    ('relu', []),
    ('sigmoid', []),
    ('tanh', []),
    ('leaky_relu', [0.1]),
    ('leaky_relu', [0.2]),
    ('elu', [0.1]),
    ('elu', [0.2]),
    ('relu6', [])])
def test_fused_convolution(seed, kernel, pad, stride, dilation, group, channel_last, decay_rate, eps, nonlinearity, nonlinearity_args, with_bias, bn, add2, ctx, func_name):

    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = list(create_inputs(rng, kernel, pad, stride,
                                dilation, group, with_bias, bn, add2, channel_last))
    base_axis = 1
    batch_stat = True if bn is None else bn
    bn_backward = [None, None, None, None]
    if bn is True:
        bn_backward = [True, True, False, False]
    elif bn is False:
        bn_backward = [True, True, True, True]
    function_tester(rng, F.fused_convolution, ref_fused_convolution,
                    inputs,
                    ref_grad=ref_grad_fused_convolution,
                    func_args=[base_axis, (pad, pad), (stride, stride), (dilation, dilation), group, channel_last,
                               decay_rate, eps, batch_stat, nonlinearity, nonlinearity_args],
                    backward=[True, True, True if with_bias else None] +
                    bn_backward + [True if add2 else None],
                    ctx=ctx, func_name=func_name, dstep=1e-2, atol_b=1e-2)
