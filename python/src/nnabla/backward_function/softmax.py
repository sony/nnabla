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


class SoftmaxBackward(BackwardFunction):

    @property
    def name(self):
        return 'SoftmaxBackward'

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
        inp0 = inputs[self._num_inputs_fwd]      # y
        inp1 = inputs[self._num_inputs_fwd + 1]  # dy
        v = nn.Variable(inp1.shape)
        v.data = inp0.data
        v.grad = inp1.data
        outputs_fwd += [v]
        return inputs_fwd, outputs_fwd

    def backward_impl(self, inputs, outputs, prop_down, accum):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        axis = self.forward_func.info.args["axis"]

        # Inputs
        x0 = inputs[0].data
        y0 = inputs[1].data
        dy = inputs[2].data
        # Outputs
        dx0 = outputs[0].data
        # Grads of inputs
        g_x0 = inputs[0].grad
        g_y0 = inputs[1].grad
        g_dy = inputs[2].grad
        # Grads of outputs
        g_dx0 = outputs[0].grad

        # w.r.t. x0
        if prop_down[0]:
            gdx_y = g_dx0 * y0
            gdx_dy_y = gdx_y * dy
            dy_y = dy * y0
            gdx_y_sum = F.sum(gdx_y, axis, True)
            dy_y_sum = F.sum(dy_y, axis, True)
            gdx_dy_y_sum = F.sum(gdx_dy_y, axis, True)

            t1 = gdx_dy_y
            t2 = y0 * gdx_dy_y_sum
            t3 = gdx_y_sum * (y0 * dy_y_sum - dy_y)
            t4 = dy_y_sum * (y0 * gdx_y_sum - gdx_y)

            g_x0_ = t1 - t2 + t3 + t4
            if accum[0]:
                g_x0 += g_x0_
            else:
                g_x0.copy_from(g_x0_)

        # w.r.t. dy
        if prop_down[2]:
            si = nn.Variable(x0.shape).apply(
                data=x0, grad=g_dy, need_grad=True)
            so = nn.Variable(dx0.shape).apply(data=y0, grad=g_dx0)
            self.forward_func.backward([si], [so], accum=[accum[2]])
