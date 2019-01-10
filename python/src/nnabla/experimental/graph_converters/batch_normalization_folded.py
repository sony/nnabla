import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

import os

from .identity import IdentityConverter
from .helpers import GraphInfo


class BatchNormalizationFoldedConverter(IdentityConverter):
    """
    Single `Convolution -> BatchNormalization` pass is folded into one `Convolution`.

    If there is a `Convolution -> BatchNormalization` pass, fold the batch normalization paramters to the kernel and bias (if it exists) of the preceeding convolution, then skip the batch normalization following the convolution.

    Args:
        black_list (list): Black list of the function list.
        params (:obj:`OrderedDict`): Result of nn.get_parameters().
        name (:obj:`str`): Prefix of the parameter scope.

    """

    def __init__(self,
                 black_list=[], params=None,
                 inner_prod_functions=None,
                 name="bn-folded"):

        import nnabla.function_bases as FB
        import nnabla as nn

        super(BatchNormalizationFoldedConverter, self).__init__(black_list,
                                                                params, name)

        self.inner_prod_functions = inner_prod_functions if inner_prod_functions \
            else ["Affine",
                  "Convolution",
                  "Deconvolution"]

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
                # TODO: error check

                # Batch normalization check, then skip
                if func.name == "BatchNormalization":
                    i0 = func.inputs[0]
                    bn_func = func
                    # Test mode check
                    if bn_func.info.args["batch_stat"] == False:
                        # `Target Func -> BN` check from BN
                        if i0.parent.info.type_name in self.inner_prod_functions:
                            nn.logger.info("{} is skipped.".format(func.name))
                            continue

                # `Target Func -> BN` conversion
                if func.name in self.inner_prod_functions:
                    inner_prod_func = func

                    o0 = inner_prod_func.outputs[0]
                    fs = self.graph_info.variable_to_funcs[o0]
                    # No branch check #TODO: branching check (really needed?)
                    if fs is not None and len(fs) == 1:
                        # `Target Func -> BN` check
                        bn_func = fs[0]
                        if bn_func.name == "BatchNormalization":

                            # Test mode check
                            if bn_func.info.args["batch_stat"] == False:

                                # Perform `Target Func -> BN` conversion
                                nn.logger.info("BatchNormalization parameters are folded to "
                                               "the preceding convolution.")
                                o = self._inner_prod_bn_conversion(
                                    inner_prod_func, bn_func)
                                continue

                # Identity conversion
                o = self._identity_conversion(func)

        self.end_variable = o
        return self.end_variable

    def _compute_folded_parameters(self, inner_prod_func, bn_func):
        # Squeeze
        beta_data = np.squeeze(bn_func.inputs[1].d)
        gamma_data = np.squeeze(bn_func.inputs[2].d)
        mean_data = np.squeeze(bn_func.inputs[3].d)
        var_data = np.squeeze(bn_func.inputs[4].d)
        eps_data = bn_func.info.args["eps"]
        # Reshape
        w = inner_prod_func.inputs[1]
        r_shape = [1 for _ in range(len(w.shape) - len(beta_data.shape))]
        beta_data = beta_data.reshape(list(beta_data.shape) + r_shape)
        gamma_data = gamma_data.reshape(list(gamma_data.shape) + r_shape)
        mean_data = mean_data.reshape(list(mean_data.shape) + r_shape)
        var_data = var_data.reshape(list(var_data.shape) + r_shape)
        sigma_data = np.sqrt(var_data + eps_data)
        # Reshape again if affine
        if inner_prod_func.name == "Affine":  # (inp, out) -> (out, inp)
            beta_data = beta_data.reshape(
                beta_data.shape[1], beta_data.shape[0])
            gamma_data = gamma_data.reshape(
                gamma_data.shape[1], gamma_data.shape[0])
            mean_data = mean_data.reshape(
                mean_data.shape[1], mean_data.shape[0])
            var_data = var_data.reshape(var_data.shape[1], var_data.shape[0])
            sigma_data = np.sqrt(var_data + eps_data)
        # Fold
        c0 = gamma_data / sigma_data
        c1 = beta_data - (gamma_data * mean_data) / sigma_data
        w_data = w.d
        w_data = c0 * w_data
        b_data = c1
        if len(inner_prod_func.inputs) == 3:
            b = inner_prod_func.inputs[2]
            b_data += c0 * b.d.reshape(b_data.shape)
        return w_data, np.squeeze(b_data)

    def _inner_prod_bn_conversion(self, inner_prod_func, bn_func):
        # Fold parameters
        w_data, b_data = self._compute_folded_parameters(
            inner_prod_func, bn_func)

        # W
        w = inner_prod_func.inputs[1]
        idx = list(self.params.values()).index(w)
        name = list(self.params.keys())[idx]
        w = nn.parameter.get_parameter_or_create(name,
                                                 w.shape,
                                                 w_data,
                                                 w.need_grad)
        # b (borrow from w)
        name = os.path.join("/".join(name.rstrip().split("/")[:-1]), "b")
        b = nn.parameter.get_parameter_or_create(name,
                                                 b_data.shape,
                                                 b_data,
                                                 need_grad=True)

        # Input conversion
        x = inner_prod_func.inputs[0]
        x = self.input_map[x] if x in self.input_map else x
        inputs = [x, w, b]

        # Function call
        o = self._call_function(inner_prod_func.name,
                                inputs,
                                inner_prod_func.info.args)

        # Map output of ref graph (BN output) to output of new graph
        o_bn = bn_func.outputs[0]
        self.input_map[o_bn] = o  # new input

        # Store output (just in case)
        self.outputs.append(o)

        return o
