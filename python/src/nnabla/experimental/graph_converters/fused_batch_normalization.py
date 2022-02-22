# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

import nnabla.parametric_functions as PF

from .graph_converter import FunctionModifier


class FusedBatchNormalizationModifier(FunctionModifier):
    """
    Block `BatchNormalization -> Add2 -> Non-Linear` pass is fused into one `FusedBatchNormalization`.

    If there is a block `BatchNormalization -> Add2 -> Non-Linear` pass,
    remove all the block functions and replace the whole block to `FusedBatchNormalization`.

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.FusedBatchNormalizationModifier()]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)
    """

    def __init__(self):
        super(FusedBatchNormalizationModifier, self).__init__()
        self._name = ''
        self._block = False
        self._bn_args = None
        self._add2_input1 = None
        self._cnt = 1
        self._fct_set = {
            'ReLU': 'relu'
        }

    def modify(self, f, inputs):
        outputs = f.outputs[0]

        # Not end
        if len(outputs.function_references) == 0:
            return

        # Check fused bn block start
        if not self._block and self._is_fused_bn_block(f, inputs):
            self._block = True

        # Remove BatchNormalization
        if self._block and f.info.type_name == 'BatchNormalization':
            self._bn_args = f.info.args
            self._name = self.get_parameter_scope(inputs[0])
            return inputs[0]

        # Remove Add2
        if self._block and f.info.type_name == 'Add2':
            self._add2_input1 = inputs[1]
            return inputs[0]

        # Remove non linear function then connect fused bn
        if self._block and f.info.type_name in self._fct_set:
            f_non_linear = self._fct_set[f.info.type_name]
            h = PF.fused_batch_normalization(
                inputs[0], self._add2_input1,
                nonlinearity=f_non_linear, **self._bn_args,
                name='fused{}-{}'.format(self._name, self._cnt))
            self._cnt += 1
            self._block = False
            return h

    def _is_fused_bn_block(self, f, inputs):
        outputs = f.outputs[0]

        # Function is BN whose next function is Add2,
        # function after Add2 is not non-linear
        next_func = outputs.function_references[0]

        if len(next_func.outputs[0].function_references) == 0:
            return False

        nnext_func = next_func.outputs[0].function_references[0]

        if f.info.type_name == 'BatchNormalization' \
           and next_func.info.type_name == 'Add2'   \
           and nnext_func.info.type_name in self._fct_set:
            return True

        return False

    def __finish__(self):
        self._name = ''
        self._block = False
        self._bn_args = None
        self._add2_input1 = None
        self._cnt = 1
