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

from nnabla.function import PythonFunction


class BackwardFunction(PythonFunction):
    """Parent class for the backward functions.
    """

    def __init__(self, ctx):
        super(BackwardFunction, self).__init__(ctx)

        self._num_inputs = 0
        self._num_outputs = 0
        self._num_inputs_fwd = 0
        self._num_outputs_fwd = 0

    def set_num_inputs_and_outputs(self, num_inputs, num_outputs, num_inputs_fwd, num_outputs_fwd):
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_inputs_fwd = num_inputs_fwd
        self._num_outputs_fwd = num_outputs_fwd

    def set_forward_function(self, f):
        self.forward_func = f

    def min_inputs(self):
        return self._num_inputs

    def min_outputs(self):
        return self._num_outputs

    def setup_impl(self, inputs, outputs):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        # Reset shape of outputs
        for i in range(self._num_inputs_fwd):
            inp = inputs[i]
            out = outputs[i]
            out.reset_shape(inp.shape, True)

    def forward_impl(self, inputs, outputs):
        # inputs: [inputs_fwd_graph] + [inputs_bwd_graph] or
        # [inputs_fwd_graph] + [outputs_fwd_graph] + [inputs_bwd_graph]

        inputs_fwd, outputs_fwd = self._create_forward_inputs_and_outputs(
            inputs, outputs)
        self.forward_func.backward(inputs_fwd, outputs_fwd, accum=[
                                   False] * self._num_inputs_fwd)
