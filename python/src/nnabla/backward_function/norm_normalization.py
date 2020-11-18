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


class NormNormalizationBackward(BackwardFunction):

    @property
    def name(self):
        return 'NormNormalizationBackward'

    def _create_forward_inputs_and_outputs(self, inputs, outputs):
        msg = ("Implement this function correctly \n"
               "if the backward function takes the output(s) of the forward function.\n"
               "See the SigmoidBackward in that case.\n"
               "Delete this error message after implementing.")
        raise Exception(msg)

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
            "The backward method of NormNormalizationBackward class is not implemented.")
