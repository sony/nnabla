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
from .backward_function import BackwardFunction


class MaxPoolingBackward(BackwardFunction):

    @property
    def name(self):
        return 'MaxPoolingBackward'

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
            # In cudnn, it seems we have to set the same pointer value
            v.data = self.forward_func.outputs[0].data
            v.grad = inp.data
            outputs_fwd += [v]
        return inputs_fwd, outputs_fwd

    def backward_impl(self, inputs, outputs, prop_down, accum):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        assert len(
            inputs[0].shape) > 3, "len(inputs[0] shape) should be greater than 3."

        # Args
        kernel = self.forward_func.info.args["kernel"]
        stride = self.forward_func.info.args["stride"]
        ignore_border = self.forward_func.info.args["ignore_border"]
        pad = self.forward_func.info.args["pad"]
        channel_last = self.forward_func.info.args["channel_last"]

        # Inputs
        x0 = inputs[0].data
        dy = inputs[1].data
        # Outputs
        dx0 = outputs[0].data
        # Grads of inputs
        g_x0 = inputs[0].grad
        g_dy = inputs[1].grad
        # Grads of outputs
        g_dx0 = outputs[0].grad

        # Compute
        ctx = nn.get_current_context()
        backward_func = nn.function.MaxPoolingBackward(
            ctx, kernel, stride, ignore_border, pad, channel_last)

        if prop_down[1]:
            x0_ = nn.Variable(x0.shape).apply(
                data=x0, grad=g_x0, need_grad=True)
            dy_ = nn.Variable(dy.shape).apply(
                data=dy, grad=g_dy, need_grad=True)
            dx0_ = nn.Variable(dx0.shape).apply(data=dx0, grad=g_dx0)
            backward_func.setup([x0_, dy_], [dx0_])
            backward_func.backward([x0_, dy_], [dx0_], accum=accum)

    def forward_impl(self, inputs, outputs):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        inputs_fwd, outputs_fwd = self._create_forward_inputs_and_outputs(
            inputs, outputs)
        self.forward_func.backward(inputs_fwd, outputs_fwd, accum=[
                                   False] * self._num_inputs_fwd)
