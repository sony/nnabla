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


from functools import partial

import nnabla as nn
import nnabla.function as _F
import nnabla.functions as F
import numpy as np
from nnabla.function import PythonFunction

from .utils import force_tuple


def double_backward_for_batch(g_dx0, g_db0, g_dg0,
                              dy, x0, b0, g0, rm, rv,
                              axes, decay_rate, eps):
    # Prerequisite
    # axes reduced and denominator
    axes0 = [a for a in range(x0.ndim)]
    axes = [a+x0.ndim*(a < 0) for a in axes]
    axes = list(set(axes0) - set(axes))
    F_sum = partial(F.sum, axis=axes, keepdims=True)
    F_mean = partial(F.mean, axis=axes, keepdims=True)
    de = np.prod([x0.shape[a] for a in axes])        # denominator
    bm = F_mean(x0)                                  # batch mean
    bv = F_mean(x0 ** 2.0) - bm ** 2.0               # batch variance
    x0_bm = x0 - bm                                  # x0 - batch mean
    v_eps_r1 = (bv + eps) ** -1.0               # (variance + eps) * (-1)
    v_eps_rsqrt1 = (bv + eps) ** (-1.0 / 2.0)   # (variance + eps) * (-1/2)
    v_eps_rsqrt3 = v_eps_rsqrt1 ** 3.0          # (variance + eps) * (-3/2)

    # common factors
    dy_x0_bm_sum = F_sum(dy * x0_bm)
    dy_sum = F_sum(dy)
    g_dx0_x0_bm_sum = F_sum(g_dx0 * x0_bm)
    g_dx0_sum = F_sum(g_dx0)

    # wrt. x
    # from dx
    dv = (-1.0 / 2.0) * g0 * v_eps_rsqrt3 * F_sum(dy * x0_bm)
    g_dx0_dy_sum = F_sum(g_dx0 * dy)

    g1 = (-1.0 / de) * v_eps_rsqrt3 * g_dx0_dy_sum * g0 * x0_bm
    g2 = (1.0 / de) * g0 * g_dx0_x0_bm_sum * v_eps_rsqrt3 * (1.0 / de * dy_sum - dy
                                                             + (3.0 / de) * v_eps_r1 * dy_x0_bm_sum * x0_bm)
    g2 += (2.0 / de) * dv * (g_dx0 - (1.0 / de) * g_dx0_sum)
    g3 = (1.0 / de ** 2.0) * g_dx0_sum * dy_sum * g0 * v_eps_rsqrt3 * x0_bm
    g_x0 = g1 + g2 + g3

    # from gamma
    t1 = (dy - dy_sum / de) * v_eps_rsqrt1
    t2 = (- 1.0 / de) * dy_x0_bm_sum * v_eps_rsqrt3 * x0_bm
    g_x0 += g_dg0 * (t1 + t2)

    # wrt. beta
    # zero, do nothing

    # wrt. gamma
    t1 = dy * v_eps_rsqrt1
    t2 = (- 1.0 / de) * dy_x0_bm_sum * v_eps_rsqrt3 * x0_bm
    t3 = (- 1.0 / de) * dy_sum * v_eps_rsqrt1
    g_g0 = F_sum(g_dx0 * (t1 + t2 + t3))

    # wrt. dy
    t1 = g_dx0 * g0 * v_eps_rsqrt1
    t2 = - (1.0 / de) * g0 * v_eps_rsqrt3 * g_dx0_x0_bm_sum * x0_bm
    t3 = - (1.0 / de) * g0 * v_eps_rsqrt1 * g_dx0_sum
    x0_hat = x0_bm * v_eps_rsqrt1
    g_dy = (t1 + t2 + t3) + g_dg0 * x0_hat + g_db0

    return g_dy, g_x0, None, g_g0


def double_backward_for_global(g_dx0, g_db0, g_dg0,
                               dy, x0, b0, g0, rm, rv,
                               axes, decay_rate, eps):
    # Prerequisite
    # axes reduced and denominator
    axes0 = [a for a in range(x0.ndim)]
    axes = list(set(axes0) - set(axes))
    # (variance + eps) * (-1/2)
    v_eps_rsqrt1 = (rv + eps) ** (-1.0 / 2.0)

    # wrt. x
    g_x0 = g_dg0 * dy * v_eps_rsqrt1

    # wrt. beta
    # zero, do nothing

    # wrt. gamma
    g_g0 = F.sum(g_dx0 * dy * v_eps_rsqrt1, axes, True)

    # no backward wrt. rm and rv

    # wrt. dy
    g_dy = g_dx0 * g0 * v_eps_rsqrt1 \
        + g_dg0 * (x0 - rm) * v_eps_rsqrt1 + g_db0

    return g_dy, g_x0, None, g_g0


class BatchNormalizationBackward(PythonFunction):

    def __init__(self, ctx, axes=[], decay_rate=0.9, eps=1e-05, batch_stat=True,
                 no_scale=False, no_bias=False):
        super(BatchNormalizationBackward, self).__init__(ctx)
        self._func = _F.BatchNormalization(
            ctx, axes, decay_rate, eps, batch_stat, no_scale, no_bias)
        self.axes = axes
        self.decay_rate = decay_rate
        self.eps = eps
        self.batch_stat = batch_stat
        self.no_scale = no_scale
        self.no_bias = no_bias

        # Variable indices
        self.dy_idx = 0
        self.x0_idx = 1
        self.b0_idx = None
        self.g0_idx = None
        self.rm_idx = None
        self.rv_idx = None
        v_idx = 2
        if not no_bias:
            self.b0_idx = v_idx
            v_idx += 1
        if not no_scale:
            self.g0_idx = v_idx
            v_idx += 1
        self.rm_idx = v_idx
        self.rv_idx = self.rm_idx + 1

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def args(self):
        return self._func.args

    def _create_fwd_inputs_outputs(self, inputs, outputs):
        # inputs_fwd: x, (beta, gamma), rmean, rvar
        # outputs_fwd: y(, bmean, bvar)
        x0 = nn.Variable(inputs[self.x0_idx].shape).apply(need_grad=True)
        b0 = nn.Variable(inputs[self.b0_idx].shape).apply(
            need_grad=True) if not self.no_bias else None
        g0 = nn.Variable(inputs[self.g0_idx].shape).apply(
            need_grad=True) if not self.no_scale else None
        rm = nn.Variable(inputs[self.rm_idx].shape).apply(need_grad=False)
        rv = nn.Variable(inputs[self.rv_idx].shape).apply(need_grad=False)
        inputs_fwd = list(
            filter(lambda v: v is not None, [x0, b0, g0, rm, rv]))
        oshape = inputs[0].shape
        outputs_fwd = [nn.Variable(oshape)]
        return inputs_fwd, outputs_fwd

    def min_inputs(self):
        return 4  # dy, x0, (b0, g0), rm, rv

    def min_outputs(self):
        return self.rm_idx - 1

    def grad_depends_output_data(self, i, o):
        return False

    def grad_depends_input_data(self, i, j):
        return True

    def setup_impl(self, inputs, outputs):
        # inputs:  dy, x, beta, gamma, rmean, rvar
        # outputs: dx, dbeta, dgamma

        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)

        self.axes = [a + inputs_fwd[0].ndim*(a < 0) for a in self.axes]

        self._func.setup(inputs_fwd, outputs_fwd)
        dx_shape = inputs_fwd[self.x0_idx - 1].shape
        db_shape = inputs_fwd[self.b0_idx -
                              1].shape if not self.no_bias else None
        dg_shape = inputs_fwd[self.g0_idx -
                              1].shape if not self.no_scale else None
        outputs[self.x0_idx - 1].reset_shape(dx_shape, True)
        outputs[self.b0_idx -
                1].reset_shape(db_shape, True) if not self.no_bias else None
        outputs[self.g0_idx -
                1].reset_shape(dg_shape, True) if not self.no_scale else None

    def forward_impl(self, inputs, outputs):
        inputs_fwd, outputs_fwd = self._create_fwd_inputs_outputs(
            inputs, outputs)

        # BN data
        x0 = inputs[self.x0_idx].data
        b0 = inputs[self.b0_idx].data if not self.no_bias else None
        g0 = inputs[self.g0_idx].data if not self.no_scale else None
        inputs_fwd[self.x0_idx - 1].data = x0
        if not self.no_bias:
            inputs_fwd[self.b0_idx - 1].data = b0
        if not self.no_scale:
            inputs_fwd[self.g0_idx - 1].data = g0
        if not self.batch_stat:
            rm = inputs[self.rm_idx].data
            rv = inputs[self.rv_idx].data
            inputs_fwd[self.rm_idx - 1].data = rm
            inputs_fwd[self.rv_idx - 1].data = rv
        # BN grad
        dx0 = outputs[self.x0_idx - 1].data
        inputs_fwd[self.x0_idx - 1].grad = dx0
        if not self.no_bias:
            db0 = outputs[self.b0_idx - 1].data
            inputs_fwd[self.b0_idx - 1].grad = db0
        if not self.no_scale:
            dg0 = outputs[self.g0_idx - 1].data
            inputs_fwd[self.g0_idx - 1].grad = dg0
        dy = inputs[0].data
        outputs_fwd[0].grad = dy
        # BN backward
        self._func.forward(inputs_fwd, outputs_fwd) \
            if self.batch_stat and not "cudnn" in self.ctx.backend else None
        self._func.backward(inputs_fwd, outputs_fwd, [False] * len(inputs_fwd))

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        g_dx0 = nn.Variable(
            outputs[self.x0_idx - 1].shape).apply(data=outputs[0].grad)
        g_db0 = nn.Variable(outputs[self.b0_idx - 1].shape).apply(data=outputs[self.b0_idx - 1].grad) \
            if not self.no_bias else 0
        g_dg0 = nn.Variable(outputs[self.g0_idx - 1].shape).apply(data=outputs[self.g0_idx - 1].grad) \
            if not self.no_scale else 0
        dy = nn.Variable(inputs[0].shape).apply(
            data=inputs[0].data, need_grad=True)
        x0 = nn.Variable(inputs[1].shape).apply(
            data=inputs[1].data, need_grad=True)
        b0 = nn.Variable(inputs[self.b0_idx].shape).apply(
            data=inputs[self.b0_idx].data, need_grad=True) if not self.no_bias else 0
        g0 = nn.Variable(inputs[self.g0_idx].shape).apply(
            data=inputs[self.g0_idx].data, need_grad=True) if not self.no_scale else 1
        rm = nn.Variable(inputs[self.rm_idx].shape).apply(
            data=inputs[self.rm_idx].data)
        rv = nn.Variable(inputs[self.rv_idx].shape).apply(
            data=inputs[self.rv_idx].data)

        double_backward = double_backward_for_batch \
            if self.batch_stat else double_backward_for_global
        with nn.auto_forward():
            g_dy_, g_x0_, g_b0_, g_g0_ = double_backward(g_dx0, g_db0, g_dg0,
                                                         dy, x0, b0, g0, rm, rv,
                                                         self.axes, self.decay_rate, self.eps)
        g_dy = inputs[0].grad
        g_x0 = inputs[1].grad
        g_g0 = inputs[self.g0_idx].grad if not self.no_scale else 0
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
        if not self.no_scale and propagate_down[self.g0_idx]:
            if accum[self.g0_idx]:
                g_g0 += g_g0_.data
            else:
                g_g0.copy_from(g_g0_.data)


def batch_normalization_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axes=(1,), decay_rate=0.9, eps=1e-05,
                                 batch_stat=True, no_scale=False, no_bias=False):
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
    # In auto-forward mode, the dynamic clear of the all inputs are
    # blocked by BatchNormalization::auto_grad_depends_input/output_data.
    ctx = nn.get_current_context()
    x_ndim = len(input_shapes[0])
    axes = [a + x_ndim * (a < 0) for a in axes]
    df = BatchNormalizationBackward(
        ctx, axes, decay_rate, eps, batch_stat, no_scale, no_bias)
    d_inputs = df(*grad_inputs, *inputs)
    return force_tuple(d_inputs) + (None, None)


def batch_normalization_backward_backward(grad_inputs, inputs, input_shapes, outputs, output_shapes, axes=(1,), decay_rate=0.9, eps=1e-05,
                                          batch_stat=True, no_scale=False, no_bias=False):
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

    axes = [a + inputs[0].ndim*(a < 0) for a in axes]

    # Align variable indices
    v_idx = 1
    if not no_bias:
        g_db0_idx = v_idx
        v_idx += 1
    if not no_scale:
        g_dg0_idx = v_idx
        v_idx += 1

    v_idx = 0
    dy_idx = v_idx
    v_idx += 1
    x0_idx = v_idx
    v_idx += 1
    if not no_bias:
        b0_idx = v_idx
        v_idx += 1
    if not no_scale:
        g0_idx = v_idx
        v_idx += 1
    rm_idx = v_idx
    rv_idx = rm_idx + 1

    # Indexing
    g_dx0 = grad_inputs[0]
    g_db0 = grad_inputs[g_db0_idx] if not no_bias else 0
    g_dg0 = grad_inputs[g_dg0_idx] if not no_scale else 0
    dy = inputs[dy_idx]
    x0 = inputs[x0_idx]
    b0 = inputs[b0_idx] if not no_bias else 0
    g0 = inputs[g0_idx] if not no_scale else 1
    rm = inputs[rm_idx]
    rv = inputs[rv_idx]

    double_backward = double_backward_for_batch \
        if batch_stat else double_backward_for_global
    g_dy, g_x0, g_b0, g_g0 = double_backward(g_dx0, g_db0, g_dg0,
                                             dy, x0, b0, g0, rm, rv,
                                             axes, decay_rate, eps)
    if not no_bias and not no_scale:
        return g_dy, g_x0, None, g_g0, None, None

    if no_bias and no_scale:
        return g_dy, g_x0, None, None
