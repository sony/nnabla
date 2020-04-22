import nnabla as nn
import numpy as np

from .identity import IdentityConverter
from .helpers import GraphInfo


class BatchNormalizationLinearConverter(IdentityConverter):
    """
    The parameters of the batch normalization replaced simple scale and bias.

    Args:
        black_list (list): Black list of the function list.
        params (:obj:`OrderedDict`): Result of nn.get_parameters().
        name (:obj:`str`): Prefix of the parameter scope.

    """

    def __init__(self,
                 black_list=[], params=None,
                 name="bn-linear"):
        super(BatchNormalizationLinearConverter, self).__init__(black_list,
                                                                params, name)

    def convert(self, vroot, entry_variables):
        """
        All functions are replaced with the same `new` function.

        Args:
            vroot (:obj:`Variable`): NNabla Variable
            entry_variables (:obj:`Variable`): Entry variable from which the conversion starts.
        """
        self.graph_info = GraphInfo(vroot)
        self.entry_variables = entry_variables

        cnt = 0
        with nn.parameter_scope(self.name):
            # Function loop in the forward order
            for t, func in enumerate(self.graph_info.funcs):
                if func.name == "BatchNormalization":
                    bn_func = func
                    # TODO: should deal with both?
                    if not bn_func.info.args["batch_stat"]:
                        o = self._bn_linear_conversion(bn_func, cnt)
                        cnt += 1
                        continue
                # Identity conversion
                o = self._identity_conversion(func)

        self.end_variable = o
        return self.end_variable

    def _bn_linear_conversion(self, bn_func, cnt):
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
        x = bn_func.inputs[0]
        x = self.input_map[x] if x in self.input_map else x

        c0 = nn.parameter.get_parameter_or_create("c0-{}-{}".format(self.name, cnt),
                                                  c0_data.shape, c0_data)
        c1 = nn.parameter.get_parameter_or_create("c1-{}-{}".format(self.name, cnt),
                                                  c1_data.shape, c1_data)

        # Function call
        o = c0 * x + c1

        # Map output of ref graph to output of new graph
        x = bn_func.outputs[0]
        self.input_map[x] = o

        # Store output (just in case)
        self.outputs.append(o)

        return o
