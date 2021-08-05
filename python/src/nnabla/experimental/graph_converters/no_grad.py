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
import nnabla.functions as F
import numpy as np

from nnabla.parameter import get_parameter_or_create
from nnabla.initializer import ConstantInitializer

from .graph_converter import FunctionModifier


class NoGradModifier(FunctionModifier):
    """
    All functions are replaced to the same `new` function.

    Args:
        inputs (:obj:`dict`): Input variable mapping from the original input to another input. Default is the empty dictionary, so the new graph shares the original inputs.

    Examples:

    .. code-block:: python

       pred = Model(...)
       x = nn.Variable(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.NoGradModifier()]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)
    """

    def __init__(self):
        super(NoGradModifier, self).__init__()

    def modify(self, f, inputs):
        params = [v.data for v in nn.get_parameters(grad_only=False).values()]
        inputs_ = []
        for inp in inputs:
            if inp.data not in params:
                inputs_.append(inp)
            else:
                inp = inp.get_unlinked_variable(need_grad=False)
                inputs_.append(inp)

        o = self._call_function(
            f.info.type_name, inputs_, f.info.args)
        return o
