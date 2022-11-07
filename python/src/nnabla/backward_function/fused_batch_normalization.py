# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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


import nnabla as nn
import nnabla.function as _F
import nnabla.functions as F
from nnabla.function import PythonFunction

from .batch_normalization import double_backward_for_batch as bn_double_backward_for_batch
from .batch_normalization import double_backward_for_global as bn_double_backward_for_global
from .utils import no_grad


def double_backward(g_dx0, g_db0, g_dg0, g_dz0,
                    dy, x0, b0, g0, rm, rv, y0, z0,
                    axes, decay_rate, eps, nonlinearity, batch_stat):
    # Factorized forward graph looks like
    # [x0, b0, g0, rm, rv] -> BN -> [u(, z0)] -> Add -> [v] -> ReLU -> [y]
    # Factorized backward graph looks like
    # [dy] -> d(ReLU) -> [dv] -> d(Add) -> [du(, dz0)] -> d(BN) -> [dx0, db0, dg0, drm, drv]

    # 1ST-ORDER
    # d(Avtivation)
    if nonlinearity == "relu":
        m0 = F.greater_scalar(y0, 0)
        m0 = no_grad(m0)
        dv = dy * m0
    elif nonlinearity == "":
        dv = dy
    # d(Add)
    du = dv

    # 2ND-ORDER
    # dd(BN)
    bn_double_backward = bn_double_backward_for_batch if batch_stat else \
        bn_double_backward_for_global
    g_du, g_x0, g_b0, g_g0 = bn_double_backward(g_dx0, g_db0, g_dg0,
                                                du, x0, b0, g0, rm, rv,
                                                axes, decay_rate, eps)
    # dd(Add)
    g_dv = g_du
    if g_dz0:
        g_dv += g_dz0
    # dd(Activation)
    if nonlinearity == "relu":
        g_dy = g_dv * m0
    elif nonlinearity == "":
        g_dy = g_dv

    return g_dy, g_x0, g_b0, g_g0


class FusedBatchNormalizationBackward(PythonFunction):

    def __init__(self, ctx, axes=[], decay_rate=0.9, eps=1e-05, batch_stat=True, nonlinearity="relu"):
        super(FusedBatchNormalizationBackward, self).__init__(ctx)
        self._func = _F.FusedBatchNormalization(
            ctx, axes, decay_rate, eps, batch_stat, nonlinearity)
        self.axes = axes
        self.decay_rate = decay_rate
        self.eps = eps
        self.batch_stat = batch_stat
        self.nonlinearity = nonlinearity
        self._is_add = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return self._func.args

    @property
    def is_add(self):
        return self._is_add

    @is_add.setter
    def is_add(self, is_add):
        self._is_add = is_add

    def _create_fwd_inputs_outputs(self, inputs, outputs):
        # inputs_fwd: x, beta, gamma, rmean, rvar(, z)
        # outputs_fwd: y(, bmean, bvar)
        x0 = nn.Variable(inputs[1].shape).apply(need_grad=True)
        b0 = nn.Variable(inputs[2].shape).apply(need_grad=True)
        g0 = nn.Variable(inputs[3].shape).apply(need_grad=True)
        rm = nn.Variable(inputs[4].shape).apply(need_grad=False)
        rv = nn.Variable(inputs[5].shape).apply(need_grad=False)
        z0 = nn.Variable(inputs[7].shape).apply(need_grad=True) \
            if self._is_add else None
        inputs_fwd = [x0, b0, g0, rm, rv, z0] if z0 else [x0, b0, g0, rm, rv]
        oshape = inputs[0].shape
        outputs_fwd = [nn.Variable(oshape)]
        return inputs_fwd, outputs_fwd

    def min_inputs(self):
        return 7

    def min_outputs(self):
        return 4 if self._is_add else 3

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return True

    def setup_impl(self, inputs, outputs):
        # inputs:  dy, x, beta, gamma, rmean, rvar, y(, z)
        # outputs: dx, dbeta, dgamma(, dz)
        # y (output of FusedBN) is needed for computing the mask for the relu backward.
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)
        self._func.setup(inputs_fwd, outputs_fwd)
        dx_shape = inputs_fwd[0].shape
        db_shape = inputs_fwd[1].shape
        dg_shape = inputs_fwd[2].shape
        dz_shape = inputs_fwd[5].shape if self._is_add else None
        outputs[0].reset_shape(dx_shape, True)
        outputs[1].reset_shape(db_shape, True)
        outputs[2].reset_shape(dg_shape, True)
        outputs[3].reset_shape(dz_shape, True) if self._is_add else None

    def forward_impl(self, inputs, outputs):
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)

        # FusedBN data
        x0 = inputs[1].data
        b0 = inputs[2].data
        g0 = inputs[3].data
        z0 = inputs[7].data if self._is_add else None
        inputs_fwd[0].data = x0
        inputs_fwd[1].data = b0
        inputs_fwd[2].data = g0
        if self._is_add:
            inputs_fwd[5].data = z0
        if not self.batch_stat:
            rm = inputs[4].data
            rv = inputs[5].data
            inputs_fwd[3].data = rm
            inputs_fwd[4].data = rv
        # FusedBN grad
        dx0 = outputs[0].data
        db0 = outputs[1].data
        dg0 = outputs[2].data

        inputs_fwd[0].grad = dx0
        inputs_fwd[1].grad = db0
        inputs_fwd[2].grad = dg0
        if self._is_add:
            dz0 = outputs[3].data
            inputs_fwd[5].grad = dz0
        dy = inputs[0].data
        outputs_fwd[0].grad = dy
        # FusedBN backward
        self._func.forward(inputs_fwd, outputs_fwd) \
            if self.batch_stat and not "cudnn" in self.ctx.backend else None
        self._func.backward(inputs_fwd, outputs_fwd, [False] * len(inputs_fwd))

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        g_dx0 = nn.Variable(outputs[0].shape).apply(data=outputs[0].grad)
        g_db0 = nn.Variable(outputs[1].shape).apply(data=outputs[1].grad)
        g_dg0 = nn.Variable(outputs[2].shape).apply(data=outputs[2].grad)
        g_dz0 = nn.Variable(outputs[3].shape).apply(data=outputs[3].grad) \
            if self._is_add else None
        dy = nn.Variable(inputs[0].shape).apply(
            data=inputs[0].data, need_grad=True)
        x0 = nn.Variable(inputs[1].shape).apply(
            data=inputs[1].data, need_grad=True)
        b0 = nn.Variable(inputs[2].shape).apply(
            data=inputs[2].data, need_grad=True)
        g0 = nn.Variable(inputs[3].shape).apply(
            data=inputs[3].data, need_grad=True)
        rm = nn.Variable(inputs[4].shape).apply(data=inputs[4].data)
        rv = nn.Variable(inputs[5].shape).apply(data=inputs[5].data)
        y0 = nn.Variable(inputs[6].shape).apply(data=inputs[6].data)
        z0 = nn.Variable(inputs[7].shape).apply(data=inputs[7].data, need_grad=True) \
            if self._is_add else None

        with nn.auto_forward():
            g_dy_, g_x0_, g_b0_, g_g0_ = double_backward(g_dx0, g_db0, g_dg0, g_dz0,
                                                         dy, x0, b0, g0, rm, rv, y0, z0,
                                                         self.axes, self.decay_rate, self.eps,
                                                         self.nonlinearity, self.batch_stat)
        g_dy = inputs[0].grad
        g_x0 = inputs[1].grad
        g_g0 = inputs[3].grad
        g_z0 = inputs[7].grad if self._is_add else None
        # wrt dy
        if propagate_down[0]:
            if accum[0]:
                g_dy += g_dy_.data
            else:
                g_dy.copy_from(g_dy_.data)
        # wrt x
        if propagate_down[1]:
            if accum[1]:
                g_x0 += g_x0_.data
            else:
                g_x0.copy_from(g_x0_.data)
        # wrt g
        if propagate_down[3]:
            if accum[3]:
                g_g0 += g_g0_.data
            else:
                g_g0.copy_from(g_g0_.data)


def fused_batch_normalization_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axes=(1,), decay_rate=0.9, eps=1e-05,
                                       batch_stat=True, nonlinearity='relu'):
    """
    Args:
      grad_inputs (list of :obj:`nnabla.Variable`): Propagated grads to this backward function.
      inputs (list of :obj:`nnabla.Variable` and None): Input Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      input_shapes (list of tuple of :obj:`int`): Input shapes of the forward function.
          The shapes of the inputs in which None is set can be passed.
      outputs (list of :obj:`nnabla.Variable` and None): Output Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      output_shapes (list of tuple of :obj:`int`): Output shapes of the forward function.
          The shapes of the outputs in which None is set can be passed.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    if nonlinearity not in ["", "relu"]:
        raise ValueError("nonlinearity must be either '' or 'relu'.")
    ctx = nn.get_current_context()
    df = FusedBatchNormalizationBackward(
        ctx, axes, decay_rate, eps, batch_stat, nonlinearity)
    dy = grad_inputs[0]
    x0 = inputs[0]
    b0 = inputs[1]
    g0 = inputs[2]
    rm = inputs[3]
    rv = inputs[4]
    z0 = inputs[5] if len(inputs) == 6 else None
    df.is_add = True if z0 else False
    y0 = outputs[0]
    if df.is_add:
        dx0, db0, dg0, dz0 = df(dy, x0, b0, g0, rm, rv, y0, z0)
        return dx0, db0, dg0, None, None, dz0
    else:
        dx0, db0, dg0 = df(dy, x0, b0, g0, rm, rv, y0)
        return dx0, db0, dg0, None, None


def fused_batch_normalization_backward_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axes=(1,), decay_rate=0.9, eps=1e-05,
                                                batch_stat=True, nonlinearity="relu"):
    """
    Args:
      grad_inputs (list of :obj:`nnabla.Variable`): Propagated grads to this backward function.
      inputs (list of :obj:`nnabla.Variable` and None): Input Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      input_shapes (list of tuple of :obj:`int`): Input shapes of the forward function.
          The shapes of the inputs in which None is set can be passed.
      outputs (list of :obj:`nnabla.Variable` and None): Output Variables of the forward function
          if this backward function depends on it. Otherwise, None is set instead.
      output_shapes (list of tuple of :obj:`int`): Output shapes of the forward function.
          The shapes of the outputs in which None is set can be passed.
      kwargs (dict of arguments): Dictionary of the corresponding function arguments.

    Return:
      list of Variable: Return the gradients wrt inputs of the corresponding function.
    """
    is_add = True if len(inputs) == 8 else False
    if is_add:
        g_dx0 = grad_inputs[0]
        g_db0 = grad_inputs[1]
        g_dg0 = grad_inputs[2]
        g_dz0 = grad_inputs[3]
        dy = inputs[0]
        x0 = inputs[1]
        b0 = inputs[2]
        g0 = inputs[3]
        rm = inputs[4]
        rv = inputs[5]
        y0 = inputs[6]
        z0 = inputs[7]
    else:
        g_dx0 = grad_inputs[0]
        g_db0 = grad_inputs[1]
        g_dg0 = grad_inputs[2]
        dy = inputs[0]
        x0 = inputs[1]
        b0 = inputs[2]
        g0 = inputs[3]
        rm = inputs[4]
        rv = inputs[5]
        y0 = inputs[6]
        z0 = None
        g_dz0 = None

    g_dy, g_x0, g_b0, g_g0 = double_backward(g_dx0, g_db0, g_dg0, g_dz0,
                                             dy, x0, b0, g0, rm, rv, y0, z0,
                                             axes, decay_rate, eps,
                                             nonlinearity, batch_stat)
    if is_add:
        return g_dy, g_x0, g_b0, g_g0, None, None, None, None
    else:
        return g_dy, g_x0, g_b0, g_g0, None, None, None
