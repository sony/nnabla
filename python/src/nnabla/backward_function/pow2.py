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

import nnabla as nn
import nnabla.functions as F
from .backward_function import BackwardFunction


class Pow2Backward(BackwardFunction):

    @property
    def name(self):
        return 'Pow2Backward'

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

    def backward_impl(self, inputs, outputs, prop_down, accum):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        # Inputs
        x0 = inputs[0].data
        x1 = inputs[1].data
        dy = inputs[2].data
        # Outputs
        dx0 = outputs[0].data
        dx1 = outputs[1].data
        # Grads of inputs
        g_x0 = inputs[0].grad
        g_x1 = inputs[1].grad
        g_dy = inputs[2].grad
        # Grads of outputs
        g_dx0 = outputs[0].grad
        g_dx1 = outputs[1].grad

        # Computation
        if prop_down[0]:
            if accum[0]:
                # factors
                x1_1 = x1 - 1.0
                # terms
                t0 = g_dx0 * (x1 * x1_1 * x0 ** (x1 - 2.0))
                t1 = g_dx1 * (x0 ** x1_1 * (x1 * F.log(x0) + 1.0))
                g_x0 += dy * (t0 + t1)
            else:
                # factors
                x1_1 = x1 - 1.0
                # terms
                t0 = g_dx0 * (x1 * x1_1 * x0 ** (x1 - 2.0))
                t1 = g_dx1 * (x0 ** x1_1 * (x1 * F.log(x0) + 1.0))
                g_x0.copy_from(dy * (t0 + t1))
        if prop_down[1]:
            if accum[1]:
                # factors
                x1_1 = x1 - 1.0
                log_x0 = F.log(x0)
                # terms
                t0 = g_dx0 * x0 ** x1_1 * (1.0 + x1 * log_x0)
                t1 = g_dx1 * (x0 ** x1 * log_x0 ** 2.0)
                g_x1 += dy * (t0 + t1)
            else:
                # factors
                x1_1 = x1 - 1.0
                log_x0 = F.log(x0)
                # terms
                t0 = g_dx0 * x0 ** x1_1 * (1.0 + x1 * log_x0)
                t1 = g_dx1 * (x0 ** x1 * log_x0 ** 2.0)
                g_x1.copy_from(dy * (t0 + t1))
        if prop_down[2]:
            if accum[2]:
                t0 = g_dx0 * x1 * x0 ** (x1 - 1.0)
                t1 = g_dx1 * x0 ** x1 * F.log(x0)
                g_dy += t0 + t1
            else:
                t0 = g_dx0 * x1 * x0 ** (x1 - 1.0)
                t1 = g_dx1 * x0 ** x1 * F.log(x0)
                g_dy.copy_from(t0 + t1)
