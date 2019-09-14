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


class PReLUBackward(BackwardFunction):

    @property
    def name(self):
        return 'PReLUBackward'

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
        base_axis = self.forward_func.info.args["base_axis"]
        # Inputs
        x = inputs[0].data
        w = inputs[1].data
        dy = inputs[2].data
        # Outputs
        dx = outputs[0].data
        dw = outputs[1].data
        # Grads of inputs
        g_x = inputs[0].grad
        g_w = inputs[1].grad
        g_dy = inputs[2].grad
        # Grads of outputs
        g_dx = outputs[0].grad
        g_dw = outputs[1].grad

        # Computation
        shared = True if w.shape == () else False
        shape = [x.shape[i] if i == base_axis else 1 for i in range(x.ndim)]
        axes = [i for i in range(x.ndim)]
        _ = axes.pop(base_axis) if not shared else None

        def reshape(v, shape):
            if shared:
                shape = [1 for _ in range(len(shape))]
            return F.reshape(v, shape)

        if prop_down[0] or prop_down[1]:
            nmask = F.less_scalar(x, 0.0)
        if prop_down[0]:
            if accum[0]:
                g_x += dy * reshape(g_dw, shape) * nmask
            else:
                g_x.copy_from(dy * reshape(g_dw, shape) * nmask)
        if prop_down[1]:
            if accum[1]:
                g_w += F.sum(dy * g_dx * nmask, axes)
            else:
                g_w.copy_from(F.sum(dy * g_dx * nmask, axes))
        if prop_down[2]:
            pmask = 1.0 - nmask
            g_dx_ = g_dx * (pmask + reshape(w, shape) * nmask) + \
                reshape(g_dw, shape) * x * nmask
            if accum[2]:
                g_dy += g_dx_
            else:
                g_dy.copy_from(g_dx_)
