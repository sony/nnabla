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


class SigmoidCrossEntropyBackward(BackwardFunction):

    @property
    def name(self):
        return 'SigmoidCrossEntropyBackward'

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
        x0 = inputs[0].data  # logits
        t0 = inputs[1].data  # labels
        dz = inputs[2].data  # grad_input
        # Outputs
        dx0 = outputs[0].data
        dt0 = outputs[1].data
        # Grads of inputs
        g_x0 = inputs[0].grad
        g_t0 = inputs[1].grad
        g_dz = inputs[2].grad
        # Grads of outputs
        g_dx0 = outputs[0].grad
        g_dt0 = outputs[1].grad

        # Computation
        ## w.r.t. x0
        if prop_down[0]:
            sigmoid = F.sigmoid(x0)
            g_x0_ = g_dx0 * dz * sigmoid * (1.0 - sigmoid)
            if accum[0]:
                g_x0 += g_x0_
            else:
                g_x0.copy_from(g_x0_)
        ## w.r.t. t0 is not required

        ## w.r.t. dz
        if prop_down[2]:
            # Instable implementation since using `/ dz`
            # g_dz_ = g_dx0 * dx0 / dz

            # x0 and t0 are of the same shape, no need to call F.sigmoid_cross_entropy
            g_dz_ = g_dx0 * (F.sigmoid(x0) - t0)
            if accum[2]:
                g_dz += g_dz_
            else:
                g_dz.copy_from(g_dz_)
