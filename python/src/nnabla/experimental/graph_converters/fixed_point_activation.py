import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from .identity import IdentityConverter
from .helpers import GraphInfo


class FixedPointActivationConverter(IdentityConverter):
    """
    All functions specified by `activation_functions` are replaced with the fixed-point quantization function. The other functions are replaced with the same `new` function.

    Args:
        black_list (list): Black list of the function list.
        params (:obj:`OrderedDict`): Result of nn.get_parameters().
        activation_functions (list of function name): Function names to be replaced. Default is ["ReLU"].
        args_fpq (`dict`): Argument into F.quantize. Default is `{"sign": True, "n": 8, "delta": 2e-4, "quantize": True}`.
        name (:obj:`str`): Prefix of the parameter scope.
    """

    def __init__(self,
                 black_list=[], params=None,
                 activation_functions=None,
                 args_fpq={"n": 8, "sign": False,
                           "delta": 2e-4, "quantize": True},
                 name="fixed-point-activation-graph"):
        import nnabla.function_bases as FB
        import nnabla as nn
        super(FixedPointActivationConverter, self).__init__(
            black_list, params, name)

        self.activation_functions = ["ReLU"] if activation_functions is None \
            else activation_functions
        self.args_fpq = args_fpq

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
            for t, func in enumerate(self.graph_info.funcs):
                # Activation check
                if func.name in self.activation_functions:
                    activation_func = func
                    o = self._fixed_point_activation_conversion(
                        activation_func)
                    continue
                # Identity conversion
                o = self._identity_conversion(func)

        self.end_variable = o
        return self.end_variable

    def _fixed_point_activation_conversion(self, activation_func):
        # Input
        x = activation_func.inputs[0]
        x = self.input_map[x] if x in self.input_map else x

        # Conversion
        n = self.args_fpq["n"]
        sign = self.args_fpq["sign"]
        delta = self.args_fpq["delta"]
        o = F.fixed_point_quantize(x, n=n, sign=sign, delta=delta)

        # Map output of ref graph to output of new graph
        x = activation_func.outputs[0]
        self.input_map[x] = o

        # Store output (just in case)
        self.outputs.append(o)

        return o
