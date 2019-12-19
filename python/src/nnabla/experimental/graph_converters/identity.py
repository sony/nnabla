import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from .helpers import GraphInfo


class IdentityConverter(object):
    """
    All functions are replaced with the same `new` function.

    Args:
        black_list (list): Black list of the function list.
        params (:obj:`OrderedDict`): Result of nn.get_parameters().
        name (:obj:`str`): Prefix of the parameter scope.

    """

    def __init__(self, black_list=[], params=None, name="identity"):
        self.graph_info = None
        self.entry_variables = None

        self.black_list = black_list
        self.params = params if params is not None else nn.get_parameters(
            grad_only=False)
        self.name = name

        self.end_variable = None
        self.outputs = []
        # output of ref graph to output of new graph (TODO: change name)
        self.input_map = {}

    def convert(self, vroot, entry_variables):
        """
        All functions are replaced with the same `new` function.

        Args:
            vroot (:obj:`Variable`): NNabla Variable
            entry_variables (:obj:`Variable`): Entry variable from which the conversion starts.
        """
        self.graph_info = GraphInfo(vroot)
        self.entry_variables = entry_variables

        with nn.parameter_scope(self.name):
            # Function loop in the forward order
            for func in self.graph_info.funcs:
                o = self._identity_conversion(func)
            self.end_variable = o
        return self.end_variable

    def _call_function(self, type_name, inputs, args):
        import nnabla.function_bases as FB
        function_expr = "FB.F.{type_name}(nn.{ctx}, **{args})".format(
            type_name=type_name,
            ctx=nn.get_current_context(),
            args=args)
        function = eval(function_expr)
        o = function(*inputs)
        return o

    def _identity_conversion(self, func):
        import nnabla.function_bases as FB
        inputs = []

        # Inputs conversion
        for i in func.inputs:
            if i in self.entry_variables:    # entry input given by user
                idx = self.entry_variables.index(i)
                inputs.append(self.entry_variables[idx])
            elif i in self.input_map:        # new input
                inputs.append(self.input_map[i])
            elif i in self.params.values():  # parameter input
                idx = list(self.params.values()).index(i)
                name = list(self.params.keys())[idx]
                i = nn.parameter.get_parameter_or_create(name,
                                                         i.shape,
                                                         i.d,
                                                         i.need_grad)
                inputs.append(i)
            else:                            # old inputs shared by the reference graph
                inputs.append(i)

        # Function Call
        o = self._call_function(func.info.type_name, inputs, func.info.args)

        # Map output of ref graph to output of new graph
        x = func.outputs[0]
        self.input_map[x] = o

        # Store output (just in case)
        self.outputs.append(o)

        return o
