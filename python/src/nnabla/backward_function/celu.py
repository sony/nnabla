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


class CELUBackward(BackwardFunction):

    @property
    def name(self):
        return 'CELUBackward'

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

        raise NotImplementedError(
            "The backward method of CELUBackward class is not implemented.")

        # Args
        ## alpha = self.forward_func.info.args["alpha"]
        ## axis = self.forward_func.info.args["axis"]

        # Inputs
        ## x0 = inputs[0].data
        ## dy = inputs[1].data
        # Outputs
        ## dx0 = outputs[0].data
        # Grads of inputs
        ## g_x0 = inputs[0].grad
        ## g_dy = inputs[1].grad
        # Grads of outputs
        ## g_dx0 = outputs[0].grad

        # Computation
        # if prop_down[0]:
        # Slice the corresponding tensors
        ##     s0 = [0 for _ in range(dy.ndim)]
        ##     s1 = [d for d in dy.shape]
        ##     s2 = [1 for _ in range(dy.ndim)]
        ##     sp = [d // 2 if i == axis else d for i, d in enumerate(dy.shape)]
        ##     sn = [d // 2 if i == axis else 0 for i, d in enumerate(dy.shape)]
        ##     dy_p = F.slice(dy, s0, sp, s2)
        ##     dy_n = F.slice(dy, sn, s1, s2)
        # Mask for the corresponding tensor
        ##     mask = F.less_equal_scalar(x0, 0.0)

        ##     g_x0_ = g_dx0 * mask * alpha * (dy_p * F.exp(x0) + dy_n * F.exp(-x0))
        # if accum[0]:
        ##         g_x0.copy_from(g_x0 + g_x0_)
        # else:
        # g_x0.copy_from(g_x0_)
        # if prop_down[1]:
        # Mask for the corresponding tensors
        ##     mask_p = F.greater_scalar(x0, 0.0)
        ##     mask_n = 1.0 - mask_p
        # w.r.t. dy_p and dy_n
        ##     g_dy_p = g_dx0 * (mask_p + mask_n * alpha * F.exp(x0))
        ##     g_dy_n = - g_dx0 * (mask_p + mask_n * alpha * F.exp(-x0))
        # Concatenate
        ##     g_dy_ = F.concatenate(*[g_dy_p, g_dy_n], axis=axis)
        # if accum[1]:
        ##         g_dy.copy_from(g_dy + g_dy_)
        # else:
        # g_dy.copy_from(g_dy_)
