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

import nnabla as nn
import numpy as np

from .graph_converter import FunctionModifier


class BatchNormalizationSelfFoldingModifier(FunctionModifier):
    """
    The parameters of the batch normalization replaced simple scale and bias.

    Args:
        name (:obj:`str`): Prefix of the parameter scope.

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.BatchNormalizationSelfFoldingModifier()]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)

    """

    def __init__(self, name='bn-self-folding'):
        super(BatchNormalizationSelfFoldingModifier, self).__init__()
        self._cnt = 0
        self._name = name

    def modify(self, f, inputs):
        if f.info.type_name != 'BatchNormalization':
            return

        with nn.parameter_scope(self._name):
            bn_func = f
            output = self._compute_self_folded_parameters(
                bn_func, inputs, self._cnt)
            self._cnt += 1
            return output

    def _compute_self_folded_parameters(self, bn_func, inputs, cnt):
        # Conversion
        eps_data = bn_func.info.args["eps"]
        beta_data = np.squeeze(bn_func.inputs[1].d)
        gamma_data = np.squeeze(bn_func.inputs[2].d)
        mean_data = np.squeeze(bn_func.inputs[3].d)
        var_data = np.squeeze(bn_func.inputs[4].d)
        sigma_data = np.sqrt(var_data + eps_data)
        c0_data = gamma_data / sigma_data
        c1_data = beta_data - (gamma_data * mean_data) / sigma_data

        # Reshape
        oshape = bn_func.inputs[1].shape
        c0_data = c0_data.reshape(oshape)
        c1_data = c1_data.reshape(oshape)

        # Inputs
        x = inputs[0]

        c0 = nn.parameter.get_parameter_or_create("c0-{}-{}".format(self._name, cnt),
                                                  c0_data.shape, c0_data)
        c1 = nn.parameter.get_parameter_or_create("c1-{}-{}".format(self._name, cnt),
                                                  c1_data.shape, c1_data)

        # Function call
        o = c0 * x + c1

        return o

    def __finish__(self):
        self._cnt = 0
        self._name = 'bn-self-folding'
