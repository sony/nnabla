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

from collections import defaultdict

import nnabla as nn
import nnabla.functions as F


class FunctionModifier(object):
    """Base class of modifiers.

    The `modify` method is called for a function with inputs in a graph topological order
    when you call the GraphConverter(<modifiers>).convert(<root variable>) method.

    """

    def __init__(self):
        super(FunctionModifier, self).__init__()
        self._map_func_inputs = defaultdict(list)  # func: [inputs]

    def init_map_func_inputs(self, f, inputs):
        self._map_func_inputs[f] = inputs

    def get_parameter_scope(self, v):
        """Get the parameter name corresponding to v

        Args:
          v (:obj:`nnabla.Variable`): NNabla Variable Object.

        Returns:
          str: Scope name

        """
        try:
            idx = list(nn.get_parameters().values()).index(v)
            scope = '/'.join(list(nn.get_parameters().keys())
                             [idx].split('/')[:-1])
        except:
            scope = ''
        return scope

    def _force_list(self, o):
        if isinstance(o, (tuple)):
            return list(o)
        if not isinstance(o, (list)):
            return [o]

    def _copy_inputs(self, inputs):
        """Shallow Copy the inputs.
        The Variable copied is different from the original once, but the actual data and grad
        are different.
        """
        inps_cp = []
        for inp in inputs:
            i = inp.get_unlinked_variable(need_grad=inp.need_grad)
            inps_cp.append(i)
        return inps_cp

    def __call__(self, f):
        # Init condition
        if not f.inputs[0].parent and f not in self._map_func_inputs:
            self._map_func_inputs[f] = self._copy_inputs(f.inputs)

        # Lookup
        inputs = self._map_func_inputs[f]

        if len(f.inputs) > 1:
            _inputs = [inp for inp in f.inputs]
            for i in range(len(inputs)):
                if not inputs[i]:  # i-th input var has not been saved
                    inputs[i] = _inputs[i]

        # Modify
        o = self.modify(f, inputs)

        if not isinstance(o, (type(None), nn.Variable)):
            raise ValueError('Return None or nn.Variable')

        # Modify as same
        if o is None:
            o = self._modify_as_same(f, inputs)
        outputs = self._force_list(o)
        self.outputs = outputs

        # Add next func and outputs (next funcs' inputs) to table
        for o0, o1 in zip(f.outputs, outputs):
            funcs = o0.function_references
            if funcs == []:
                return
            for func in funcs:
                if len(self._map_func_inputs[func]) == 0:  # init to None
                    self._map_func_inputs[func] = [None] * len(func.inputs)
                for i, inp in enumerate(func.inputs):
                    if o0 == inp:
                        self._map_func_inputs[func][i] = o1

    def modify(self, f, inputs):
        """Modify the function.

        Implement this method in a sub class to modify a function.

        Examples:

        .. code-block:: python

           class ReLUToLeakyReLUModifier(FunctionModifier):

             def __init__(self):
               super(ReLUToLeakyReLUModifier, self).__init__()

             def modify(self, f, inputs):
               if f.info.type_name == 'ReLU':
                 x = inputs[0]
                 return F.leaky_relu(x)

        This examples is a simple case since the network topological order does not change.
        In GraphConverter, we expect the modify method is called along
        the original network tolopogical order not the modified order.
        In such a complex case, see themodify method of :obj:`BatchNormalizationFoldingModifierInner`
        as a reference.

        Args:
          f (:obj:`nnabla.function.Function`): NNabla function object.
          inputs (list of :obj:`Variable`): New inputs to `f`. This may be modified one or the same as f.inputs.

        Returns:
          :obj:`Variable` or list of :obj:`Variable`.
        """
        pass

    def _modify_as_same(self, f, inputs):
        o = self._call_function(f.info.type_name, inputs, f.info.args)
        return o

    def _call_function(self, type_name, inputs, args):
        import nnabla.function_bases as FB
        function_expr = 'FB.F.{type_name}(nn.{ctx}, **{args})'.format(
            type_name=type_name,
            ctx=nn.get_current_context(),
            args=args)
        function = eval(function_expr)
        o = function(*inputs, auto_forward=False)

        return o

    def __finish__(self):
        """Finish the very time function modification.

        Implement this method in a sub class if necessary.
        Clean up the sub class specified states, resources etc.
        It will be called by GraphConverter class instance when one time of conversion finished.

        Args:
          None

        Returns:
          None
        """
        pass

    def finish_up(self):
        """Finish the very time function modification.

        Clean up the internal modifier states.

        Args:
          None

        Returns:
          None
        """
        self._map_func_inputs = defaultdict(list)  # func: [inputs]
        self.__finish__()


class GraphConverter(object):
    """GraphConverter

    Convert a graph with the modifiers by traversing from output variables.
    """

    def __init__(self, modifiers=[]):
        self._modifiers = self._prepare_modifiers(modifiers)

    def _prepare_modifiers(self, modifiers=[]):
        mfs = []
        for modifier in modifiers:
            if isinstance(modifier, (list)):
                mfs += self._prepare_modifiers(modifier)
            else:
                mfs += [modifier]
        return mfs

    def _force_list(self, o):
        if isinstance(o, (tuple)):
            return list(o)
        if not isinstance(o, (list)):
            return [o]
        return o

    def convert(self, o):
        """
        Args:
            o (list of :obj:`nnabla.Variable`): Output variables.
        """
        oo = o
        outputs = self._force_list(o)
        # TODO: multiple inputs
        for modifier in self._modifiers:
            o = F.sink(*outputs)
            o.visit(modifier)
            outputs = modifier.outputs[0].parent.inputs
            modifier.finish_up()
        if isinstance(oo, list):
            return outputs
        return outputs[0]
