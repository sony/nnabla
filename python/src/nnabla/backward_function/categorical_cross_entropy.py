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


class CategoricalCrossEntropyBackward(BackwardFunction):

    @property
    def name(self):
        return 'CategoricalCrossEntropyBackward'

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

        # Args
        axis = self.forward_func.info.args["axis"]

        # Inputs
        y0 = inputs[0].data
        dz = inputs[2].data
        # Outputs
        dy0 = outputs[0].data
        # Grads of inputs
        g_y0 = inputs[0].grad
        g_dz = inputs[2].grad
        # Grads of outputs
        g_dy0 = outputs[0].grad

        # Computation
        if prop_down[0]:
            if accum[0]:
                g_y0 += g_dy0 * dz * F.pow_scalar(y0, -2.0)
            else:
                g_y0.copy_from(g_dy0 * dz * F.pow_scalar(y0, -2.0))
        if prop_down[2]:
            if accum[2]:
                g_dz -= F.sum(g_dy0 * F.pow_scalar(y0, -1.0), axis, True)
            else:
                g_dz.copy_from(- F.sum(g_dy0 *
                                       F.pow_scalar(y0, -1.0), axis, True))
