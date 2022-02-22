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

from .graph_converter import FunctionModifier


class RemoveFunctionModifier(FunctionModifier):
    """
    Remove specified function layer(s) from a graph.

    A convenient converter when one or more functions in
    an existing graph needs to be removed. This converter
    remove specified function(s) without recreating a new
    graph from scratch.

    Args:
        rm_funcs (list of :obj:`str`): list of function name

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.RemoveFunctionModifier(rm_funcs=['BatchNormalization', 'MulScalar'])]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)

    """

    def __init__(self, rm_funcs=[]):
        super(RemoveFunctionModifier, self).__init__()
        self._rm_funcs = rm_funcs

    def modify(self, f, inputs):
        if f.info.type_name in self._rm_funcs:
            # Get next func and inputs[0]
            next_func = f.outputs[0].function_references[0]
            inp = next_func.inputs[0]

            # BWINA:
            # Map inputs to next func for the very beginning function when its inputs[0] is 'x'
            if f.rank == 0 and inp.name == 'x':
                self._map_func_inputs[next_func].append(inp)

            return inputs[0]

    def __finish__(self):
        pass
