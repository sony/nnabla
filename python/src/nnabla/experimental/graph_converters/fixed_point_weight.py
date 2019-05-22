import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from .identity import IdentityConverter
from .helpers import GraphInfo


class FixedPointWeightConverter(IdentityConverter):
    """
    All functions specified by `inner_prod_functions` are replaced with the fixed-point counter-part. The other functions are replaced with the same `new` function.

    Args:
        black_list (list): Black list of the function list.
        params (:obj:`OrderedDict`): Result of nn.get_parameters().
        inner_prod_functions (list of function name): Function names to be replaced. Default is ["Affine", "Convolution", "Deconvolution"].
        call_forward (:obj:`bool`): Call forward function to obtain `W_q`. Default is "True", so ones do not need to call the forward function to synch quantized weights. Take care that if the network contains the batch normalization or like other normalization which computes running stats (e.g., a running mean and variance), these stats can not help being updated by this `call_forward`. To avoid that, change the argument `batch_stat` of the batch normalization layer to `False` when using this `call_foward` option `True`.
        floor (:obj:`bool`): When computing the step size, it is coerced to be the power-of-2 by using either :math:`2^ceil(log_2(abs(W)_max / (2^n - 1)))` or :math:`2^floor(log_2(abs(W)_max / (2^n - 1)))`. Default is `False`.
        args_fpq (`dict`): Argument into F.quantize. Default is `{"sign_w": True, "n_w": 8, "delta_w": 2e-4, "quantize_w": True, "sign_b": True, "n_b": 8, "delta_b": 2e-4, "quantize_b": True}`
        name (:obj:`str`): Prefix of the parameter scope.
    """

    def __init__(self,
                 black_list=[], params=None,
                 inner_prod_functions=None,
                 call_forward=True,
                 floor=False,
                 args_fpq={"sign_w": True, "n_w": 8, "delta_w": 2e-4, "quantize_w": True,
                           "sign_b": True, "n_b": 8, "delta_b": 2e-4, "quantize_b": True},
                 name="fixed-point-weight-graph"):
        super(FixedPointWeightConverter, self).__init__(
            black_list, params, name)

        self.call_forward = call_forward
        self.round_func = np.ceil if not floor else np.floor
        self.args_fpq = args_fpq

        self.inner_prod_functions = ["Affine",
                                     "Convolution",
                                     "Deconvolution"] if inner_prod_functions is None \
            else inner_prod_functions

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
                if func.name in self.inner_prod_functions:
                    inner_prod_func = func
                    o = self._fixed_point_weight_conversion(inner_prod_func)
                    continue
                # Identity conversion
                o = self._identity_conversion(func)

        self.end_variable = o

        if self.call_forward:
            o.forward(clear_buffer=True)
        return self.end_variable

    def _fixed_point_weight_conversion(self, inner_prod_func):
        # Input
        w_init = inner_prod_func.inputs[1].d
        b_init = inner_prod_func.inputs[2].d if len(
            inner_prod_func.inputs) == 3 else None
        x = inner_prod_func.inputs[0]
        x = self.input_map[x] if x in self.input_map else x

        # Quantization params
        sign_w = self.args_fpq["sign_w"]
        n_w = self.args_fpq["n_w"]
        delta_w = self.args_fpq["delta_w"]
        sign_b = self.args_fpq["sign_b"]
        n_b = self.args_fpq["n_b"]
        quantize_b = self.args_fpq["quantize_b"]
        delta_b = self.args_fpq["delta_b"]

        # Determine delta
        if not "delta_w" in self.args_fpq:
            n = n_w - 1 if sign_w is True else n_w
            w_abs_max = np.sort(np.abs(w_init.flatten()))[-1]
            delta_w = 2 ** np.round_func(np.log2(w_abs_max / (2**n - 1)))

        if not "delta_b" in self.args_fpq and len(inner_prod_func.inputs) == 3:
            n = n_b - 1 if sign_b is True else n_b
            b_abs_max = np.sort(np.abs(b_init.flatten()))[-1]
            if b_abs_max != 0:
                delta_b = 2 ** np.round_func(np.log2(b_abs_max / (2**n - 1)))
            else:
                delta_b = 0

        # Parameter name
        w = inner_prod_func.inputs[1]
        idx = list(self.params.values()).index(w)
        name = list(self.params.keys())[idx].rstrip("W/")

        # Bias
        with_bias = True if len(inner_prod_func.inputs) == 3 else False

        # Conversion
        if inner_prod_func.name == "Affine":
            omaps = w_init.shape[1]
            args = inner_prod_func.info.args
            o = PF.fixed_point_quantized_affine(x, omaps, with_bias=with_bias,
                                                w_init=w_init, b_init=b_init,
                                                sign_w=sign_w, n_w=n_w, delta_w=delta_w,
                                                sign_b=sign_b, n_b=n_b, delta_b=delta_b,
                                                name=name,
                                                **args)

        if inner_prod_func.name == "Convolution":
            omaps = w_init.shape[0]
            kernel = w_init.shape[2:]
            args = inner_prod_func.info.args
            if args.get('channel_last', False):
                raise ValueError(
                    'channel_last=True is not supported in fixed_point_quantized_convolution.')
            del args['channel_last']

            o = PF.fixed_point_quantized_convolution(x, omaps, kernel, with_bias=with_bias,
                                                     w_init=w_init, b_init=b_init,
                                                     sign_w=sign_w, n_w=n_w, delta_w=delta_w,
                                                     sign_b=sign_b, n_b=n_b, delta_b=delta_b,
                                                     name=name,
                                                     **args)

        if inner_prod_func.name == "Deconvolution":
            omaps = w_init.shape[0]
            kernel = w_init.shape[2:]
            args = inner_prod_func.info.args
            o = PF.fixed_point_quantized_deconvolution(x, omaps, kernel, with_bias=with_bias,
                                                       w_init=w_init, b_init=b_init,
                                                       sign_w=sign_w, n_w=n_w, delta_w=delta_w,
                                                       sign_b=sign_b, n_b=n_b, delta_b=delta_b,
                                                       name=name,
                                                       **args)

        # Map output of ref graph to output of new graph
        x = inner_prod_func.outputs[0]
        self.input_map[x] = o

        # Store output (just in case)
        self.outputs.append(o)

        return o
