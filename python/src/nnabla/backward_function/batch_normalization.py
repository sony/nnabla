# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

import numpy as np
import nnabla as nn
import nnabla.functions as F
from .backward_function import BackwardFunction


class BatchNormalizationBackward(BackwardFunction):

    @property
    def name(self):
        return 'BatchNormalizationBackward'

    def _create_forward_inputs_and_outputs(self, inputs, outputs):
        # Inputs on the forward graph
        inputs_fwd = []
        for i in range(self._num_inputs_fwd):
            need_grad = self.forward_func.inputs[i].need_grad
            v = nn.Variable(inputs[i].shape, need_grad=need_grad)
            v.data = inputs[i].data
            v.grad = outputs[i].data
            inputs_fwd += [v]
        # Outputs on the forward graph
        outputs_fwd = []
        for i in range(self._num_outputs_fwd):
            inp = inputs[self._num_inputs_fwd + i]
            v = nn.Variable(inp.shape)
            v.grad = inp.data
            outputs_fwd += [v]
        return inputs_fwd, outputs_fwd

        self.inputs_fwd, self.outputs_fwd = self._create_forward_inputs_and_outputs(
            inputs, outputs)
        self.forward_func.backward(self.inputs_fwd, self.outputs_fwd, accum=[
                                   False] * self._num_inputs_fwd)

    def backward_impl(self, inputs, outputs, prop_down, accum):
        # TODO: output_stat (can not be obtained by self.forward_func.info.args["batch_stat"])

        batch_stat = self.forward_func.info.args["batch_stat"]
        if batch_stat:
            self.backward_impl_for_batch(inputs, outputs, prop_down, accum)
        else:
            self.backward_impl_global(inputs, outputs, prop_down, accum)

    def backward_impl_global(self, inputs, outputs, prop_down, accum):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        # Args
        axes = self.forward_func.info.args["axes"]
        decay_rate = self.forward_func.info.args["decay_rate"]
        eps = self.forward_func.info.args["eps"]

        # TODO: factorize more
        # Inputs
        x0 = inputs[0].data  # input
        b0 = inputs[1].data  # beta
        g0 = inputs[2].data  # gamma
        rm = inputs[3].data  # running mean
        rv = inputs[4].data  # running variance
        dy = inputs[5].data  # grad input
        # Outputs
        dx0 = outputs[0].data
        db0 = outputs[1].data
        dg0 = outputs[2].data
        # Grads of inputs
        g_x0 = inputs[0].grad
        g_b0 = inputs[1].grad
        g_g0 = inputs[2].grad
        g_rm = inputs[3].grad
        g_rv = inputs[4].grad
        g_dy = inputs[5].grad
        # Grads of outputs
        g_dx0 = outputs[0].grad
        g_db0 = outputs[1].grad
        g_dg0 = outputs[2].grad

        # Prerequisite
        # axes reduced and denominator
        axes0 = [a for a in range(x0.ndim)]
        axes = list(set(axes0) - set(axes))
        # (variance + eps) * (-1/2)
        v_eps_rsqrt1 = (rv + eps) ** (-1.0 / 2.0)

        # w.r.t. x
        if prop_down[0]:
            g_x0_ = g_dg0 * dy * v_eps_rsqrt1
            if accum[0]:
                g_x0 += g_x0_
            else:
                g_x0.copy_from(g_x0_)

        # w.r.t. beta
        # zero, do nothing

        # w.r.t. gamma
        if prop_down[2]:
            g_g0_ = F.sum(g_dx0 * dy * v_eps_rsqrt1, axes, True)
            if accum[2]:
                g_g0 += g_g0_
            else:
                g_g0.copy_from(g_g0_)

        # no backward w.r.t. rm and rv

        # w.r.t. dy
        if prop_down[5]:
            g_dy_ = g_dx0 * g0 * v_eps_rsqrt1 + \
                g_dg0 * (x0 - rm) * v_eps_rsqrt1 + g_db0
            if accum[5]:
                g_dy += g_dy_
            else:
                g_dy.copy_from(g_dy_)

    def backward_impl_for_batch(self, inputs, outputs, prop_down, accum):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        # Args
        axes = self.forward_func.info.args["axes"]
        decay_rate = self.forward_func.info.args["decay_rate"]
        eps = self.forward_func.info.args["eps"]

        # TODO: factorize more
        # Inputs
        x0 = inputs[0].data  # input
        b0 = inputs[1].data  # beta
        g0 = inputs[2].data  # gamma
        rm = inputs[3].data  # running mean
        rv = inputs[4].data  # running variance
        dy = inputs[5].data  # grad input
        # Outputs
        dx0 = outputs[0].data
        db0 = outputs[1].data
        dg0 = outputs[2].data
        # Grads of inputs
        g_x0 = inputs[0].grad
        g_b0 = inputs[1].grad
        g_g0 = inputs[2].grad
        g_rm = inputs[3].grad
        g_rv = inputs[4].grad
        g_dy = inputs[5].grad
        # Grads of outputs
        g_dx0 = outputs[0].grad
        g_db0 = outputs[1].grad
        g_dg0 = outputs[2].grad

        # Prerequisite
        # axes reduced and denominator
        axes0 = [a for a in range(x0.ndim)]
        axes = list(set(axes0) - set(axes))
        de = np.prod([x0.shape[a] for a in axes])        # denominator
        bm = F.mean(x0, axes, True)                      # batch mean
        bv = F.mean(x0 ** 2.0, axes, True) - bm ** 2.0   # batch variance
        x0_bm = x0 - bm                                  # x0 - batch mean
        # (variance + eps) * (-1)
        v_eps_r1 = (bv + eps) ** -1.0
        # (variance + eps) * (-1/2)
        v_eps_rsqrt1 = (bv + eps) ** (-1.0 / 2.0)
        # (variance + eps) * (-3/2)
        v_eps_rsqrt3 = v_eps_rsqrt1 ** 3.0

        # common factors
        if prop_down[0] or prop_down[2]:
            dy_x0_bm_sum = F.sum(dy * x0_bm, axes, True)
            dy_sum = F.sum(dy, axes, True)
        if prop_down[0] or prop_down[5]:
            g_dx0_x0_bm_sum = F.sum(g_dx0 * x0_bm, axes, True)
            g_dx0_sum = F.sum(g_dx0, axes, True)

        # w.r.t. x
        if prop_down[0]:
            # from dx
            dv = (-1.0 / 2.0) * g0 * v_eps_rsqrt3 * \
                  F.sum(dy * x0_bm, axes, True)
            g_dx0_dy_sum = F.sum(g_dx0 * dy, axes, True)

            g1 = (-1.0 / de) * v_eps_rsqrt3 * g_dx0_dy_sum * g0 * x0_bm
            g2 = (1.0 / de) * g0 * g_dx0_x0_bm_sum * v_eps_rsqrt3 * (1.0 / de * dy_sum - dy
                                                                     + (3.0 / de) * v_eps_r1 * dy_x0_bm_sum * x0_bm)
            g2 += (2.0 / de) * dv * (g_dx0 - (1.0 / de) * g_dx0_sum)
            g3 = (1.0 / de ** 2.0) * g_dx0_sum * \
                dy_sum * g0 * v_eps_rsqrt3 * x0_bm

            g_x0_ = g1 + g2 + g3

            # from gamma
            t1 = (dy - dy_sum / de) * v_eps_rsqrt1
            t2 = (- 1.0 / de) * dy_x0_bm_sum * v_eps_rsqrt3 * x0_bm
            g_x0_ += g_dg0 * (t1 + t2)

            if accum[0]:
                g_x0 += g_x0_
            else:
                g_x0.copy_from(g_x0_)

        # w.r.t. beta
        # zero, do nothing

        # w.r.t. gamma
        if prop_down[2]:
            t1 = dy * v_eps_rsqrt1
            t2 = (- 1.0 / de) * dy_x0_bm_sum * v_eps_rsqrt3 * x0_bm
            t3 = (- 1.0 / de) * dy_sum * v_eps_rsqrt1
            g_g0_ = F.sum(g_dx0 * (t1 + t2 + t3), axes, True)
            if accum[2]:
                g_g0 += g_g0_
            else:
                g_g0.copy_from(g_g0_)

        # no backward w.r.t. rm and rv

        # w.r.t. dy
        if prop_down[5]:
            t1 = g_dx0 * g0 * v_eps_rsqrt1
            t2 = - (1.0 / de) * g0 * v_eps_rsqrt3 * g_dx0_x0_bm_sum * x0_bm
            t3 = - (1.0 / de) * g0 * v_eps_rsqrt1 * g_dx0_sum
            x0_hat = x0_bm * v_eps_rsqrt1
            g_dy_ = (t1 + t2 + t3) + g_dg0 * x0_hat + g_db0
            if accum[5]:
                g_dy += g_dy_
            else:
                g_dy.copy_from(g_dy_)
