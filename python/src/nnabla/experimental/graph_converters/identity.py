# Copyright 2018,2019,2020,2021 Sony Corporation.
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


class IdentityModifier(FunctionModifier):
    """
    All functions are replaced to the same `new` function.

    Args:
        inputs (:obj:`dict`): Input variable mapping from the original input to another input. Default is the empty dictionary, so the new graph shares the original inputs.

    Examples:

    .. code-block:: python

       pred = Model(...)
       x = nn.Variable(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.IdentityModifier({x0: x1})]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)
    """

    def __init__(self, inputs={}, copy_value=False):
        super(IdentityModifier, self).__init__()
        self._input_dict = inputs
        self._copy_value = copy_value

    def modify(self, f, inputs):
        # Replace only the initial inputs
        if inputs[0].parent:
            return

        # Check if replacement dict empty
        if not self._input_dict:
            return

        if f.inputs[0] in self._input_dict:
            inp_repl = self._input_dict[f.inputs[0]]
            if not self._copy_value:
                inps = inp_repl
            else:
                if inp_repl.shape != f.inputs[0].shape:
                    raise ValueError("Shape between the replaced input ({}) and original input ({}) differs when copy_value=True".format(
                        inp_repl.shape, f.inputs[0].shape))
                inp_repl.d = f.inputs[0].d.copy()
                inp_repl.g = f.inputs[0].g.copy()
            inps = [inp_repl] + inputs[1:]

            self.init_map_func_inputs(f, inps)

            o = self._call_function(
                f.info.type_name, inps, f.info.args)
            return o

    def __finish__(self):
        self._input_dict = None
        self._copy_value = False
