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


class AffineBackward(BackwardFunction):

    @property
    def name(self):
        return 'AffineBackward'

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
        with_bias = True if len(inputs) == 4 else False
        base_axis = self.forward_func.info.args["base_axis"]

        # Inputs
        x0 = inputs[0].data
        w0 = inputs[1].data
        b0 = inputs[2].data if with_bias else None
        dy = inputs[3].data if with_bias else inputs[2].data
        # Outputs
        dx0 = outputs[0].data
        dw0 = outputs[1].data
        db0 = outputs[2].data if with_bias else None
        # Grads of inputs
        g_x0 = inputs[0].grad
        g_w0 = inputs[1].grad
        g_b0 = inputs[2].grad if with_bias else None
        g_dy = inputs[3].grad if with_bias else inputs[2].grad
        # Grads of outputs
        g_dx0 = outputs[0].grad
        g_dw0 = outputs[1].grad
        g_db0 = outputs[2].grad if with_bias else None

        # Computation
        ## w.r.t. x or w.r.t. w
        if prop_down[0] or prop_down[1]:
            # we can re-use the backward of the forward with different inputs
            inp_x = nn.Variable(x0.shape).apply(
                data=g_dx0, grad=g_x0, need_grad=prop_down[0])
            inp_w = nn.Variable(w0.shape).apply(
                data=g_dw0, grad=g_w0, need_grad=prop_down[1])
            out_y = nn.Variable(dy.shape).apply(grad=dy)
            inputs = [inp_x, inp_w]
            outputs = [out_y]
            if with_bias:
                inp_b = nn.Variable(b0.shape).apply(need_grad=False)
                inputs += [inp_b]
            self.forward_func.backward(inputs, outputs, accum)
        ## w.r.t. b
        if with_bias and prop_down[2] and not accum[2]:
            zeros = F.constant(0, b0.shape)
            if not nn.get_auto_forward():
                zeros.forward()
            g_b0.copy_from(zeros.data)
        ## w.r.t. dy
        if (not with_bias and prop_down[2]) or (with_bias and prop_down[3]):
            accum_dy = accum[3] if with_bias else accum[2]
            g_dy_ = F.affine(g_dx0, w0, None, base_axis) + \
                F.affine(x0, g_dw0, None, base_axis)
            if with_bias:
                nshape = [1] * base_axis + list(b0.shape)
                g_db0 = F.reshape(g_db0, nshape)
                g_dy_ += g_db0
            if accum_dy:
                g_dy += g_dy_
            else:
                g_dy.copy_from(g_dy_)
