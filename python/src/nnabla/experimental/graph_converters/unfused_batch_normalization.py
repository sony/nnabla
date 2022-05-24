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

import nnabla.functions as F
import nnabla.parametric_functions as PF

from .graph_converter import FunctionModifier


class UnfusedBatchNormalizationModifier(FunctionModifier):
    """
    Unfuse `FusedBatchNormalization` to `BatchNormalization -> Add2 -> Non-Linear` block.

    If there is a `FusedBatchNormalization` pass, remove the fused batch normalization
    and replace it with the block `BatchNormalization -> Add2 -> Non-Linear`.

    Supported Non-Linear functions: `relu`

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.UnfusedBatchNormalizationModifier()]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)
    """

    def __init__(self):
        super(UnfusedBatchNormalizationModifier, self).__init__()
        self._fct_set = {
            'relu': F.relu
        }

    def modify(self, f, inputs):
        if f.info.type_name != 'FusedBatchNormalization':
            return

        # Prepare BN args from FBN args
        args = f.info.args
        f_non_linear = args['nonlinearity']
        del args['nonlinearity']

        # Replace FBN to "BN -> Add2 -> Non-linear"
        h = PF.batch_normalization(inputs[0], **args)
        return self._fct_set[f_non_linear](h) if len(inputs) == 5 else self._fct_set[f_non_linear](inputs[5] + h)
