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


class BatchMatmulBackward(BackwardFunction):

    @property
    def name(self):
        return 'BatchMatmulBackward'

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

        shape_a = inputs[0].shape
        shape_b = inputs[1].shape
        if shape_a[:-2] != shape_b[:-2]:
            raise ValueError("shape_a[:-2] () != shape_b[:-2] (). \n"
                             "Implicit broadcast is supported now.",
                             shape_a[:-2] != shape_b[:-2])

        # Args
        transpose_a = self.forward_func.info.args["transpose_a"]
        transpose_b = self.forward_func.info.args["transpose_b"]
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
            # condition
            if (transpose_a, transpose_b) == (True, True):
                g_x0_ = F.batch_matmul(g_dx1, dy, True, True)
            if (transpose_a, transpose_b) == (True, False):
                g_x0_ = F.batch_matmul(g_dx1, dy, False, True)
            if (transpose_a, transpose_b) == (False, True):
                g_x0_ = F.batch_matmul(dy, g_dx1, False, False)
            if (transpose_a, transpose_b) == (False, False):
                g_x0_ = F.batch_matmul(dy, g_dx1, False, True)
            # reshape for batch axes
            if g_x0_.shape != g_x0.shape:
                g_x0_ = F.reshape(g_x0_, g_x0.shape)
            if accum[0]:
                g_x0 += g_x0_
            else:
                g_x0.copy_from(g_x0_)
        if prop_down[1]:
            # condition
            if (transpose_a, transpose_b) == (True, True):
                g_x1_ = F.batch_matmul(dy, g_dx0, True, True)
            if (transpose_a, transpose_b) == (True, False):
                g_x1_ = F.batch_matmul(g_dx0, dy, False, False)
            if (transpose_a, transpose_b) == (False, True):
                g_x1_ = F.batch_matmul(dy, g_dx0, True, False)
            if (transpose_a, transpose_b) == (False, False):
                g_x1_ = F.batch_matmul(g_dx0, dy, True, False)
            # reshape for batch axes
            if g_x1_.shape != g_x1.shape:
                g_x1_ = F.reshape(g_x1_, g_x1.shape)
            if accum[1]:
                g_x1 += g_x1_
            else:
                g_x1.copy_from(g_x1_)
        if prop_down[2]:
            t1 = F.batch_matmul(g_dx0, x1, transpose_a, transpose_b)
            t2 = F.batch_matmul(x0, g_dx1, transpose_a, transpose_b)
            g_dy_ = t1 + t2
            if accum[2]:
                g_dy += g_dy_
            else:
                g_dy.copy_from(g_dy_)
