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
import nnabla.functions as F
import numpy as np
from nnabla.initializer import ConstantInitializer
from nnabla.parameter import get_parameter_or_create

from .graph_converter import FunctionModifier


class BatchNormBase(object):
    def __init__(self):
        super(BatchNormBase, self).__init__()
        self._fct_set = {
            'Affine': F.affine,
            'Convolution': F.convolution,
            'Deconvolution': F.deconvolution
        }

    def connect(self, f, x, w, b):
        fct = self._fct_set[f.info.type_name]
        h = fct(x, w, b, **f.info.args)
        return h


class AddBiasModifier(FunctionModifier, BatchNormBase):
    """
    Add bias to `Convolution` in BatchNormalization folding case if it doesn't have bias.

    Supported folding functions: `Convolution`, `Deconvolution`, `Affine`.

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.AddBiasModifier()]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)

    """

    def __init__(self):
        super(AddBiasModifier, self).__init__()

    def modify(self, f, inputs):
        fname = f.info.type_name
        if not fname in self._fct_set:
            return

        # Next or Previous func is not BatchNorm
        next_func = f.outputs[0].function_references[0]
        prev_func = f.inputs[0].parent
        if (prev_func == None
                or prev_func.info.type_name != 'BatchNormalization') \
                and next_func.info.type_name != 'BatchNormalization':
            return

        x = inputs[0]
        w = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None

        if b is not None:
            return

        scope = self.get_parameter_scope(w)
        n_outmaps = w.shape[1] if fname == 'Affine' else w.shape[0]
        with nn.parameter_scope(scope):
            b = get_parameter_or_create(
                'b', (n_outmaps, ), ConstantInitializer(), True, True)
        h = self.connect(f, x, w, b)
        return h


class BatchNormalizationFoldingModifierInner(FunctionModifier, BatchNormBase):
    """
    Single `Convolution -> BatchNormalization` pass is folded into one `Convolution`.

    If there is a `Convolution -> BatchNormalization` pass,
    fold the batch normalization parameters to the kernel
    and bias (if it exists) of the preceding convolution,
    then skip the batch normalization following the convolution.

    Supported folding functions: `Convolution`, `Deconvolution`, `Affine`.

    """

    def __init__(self, channel_last=False):
        super(BatchNormalizationFoldingModifierInner, self).__init__()
        self._channel_last = channel_last

    def modify(self, f, inputs):
        outputs = f.outputs[0]

        # Not end
        if len(outputs.function_references) == 0:
            return

        # Function is BN whose previous function is an inner-prod layer
        if f.info.type_name == 'BatchNormalization' \
                and inputs[0].parent.info.type_name in self._fct_set:
            return inputs[0]

        # Next func is not BatchNorm
        next_func = outputs.function_references[0]
        if next_func.info.type_name != 'BatchNormalization':
            return

        # Variable is not forked
        if len(f.outputs[0].function_references) != 1:
            return

        # Function is inner-product layer
        if not f.info.type_name in self._fct_set:
            return

        ip_func = f
        bn_func = next_func
        w_data, b_data = self._compute_folded_parameters(ip_func, bn_func)

        x = inputs[0]

        scope = self.get_parameter_scope(inputs[1])
        with nn.parameter_scope(scope):
            w = get_parameter_or_create('w-folded',
                                        inputs[1].shape, w_data,
                                        inputs[1].need_grad)
            b = get_parameter_or_create('b-folded',
                                        inputs[2].shape, b_data,
                                        inputs[2].need_grad)

        h = self.connect(f, x, w, b)

        return h

    def reshape_bn_parameters(self, ip_func, bn_parameters, channel_last):
        """
                                Weight layout
        conv(channel_first)     o,i,k1,k2,...kn
        conv(channel_last)      o,k1,k2,...,kn,i
        deconv(channel_first)   i,o,k1,k2,...,kn
        deconv(channel_last)    i,k1,k2,...,kn,o
        """
        if ip_func.info.type_name == 'Convolution':
            axes = list(range(bn_parameters[0].ndim))
            axis_to_switch = -1 if channel_last else 1
            axes[0], axes[axis_to_switch] = axes[axis_to_switch], axes[0]
            bn_parameters = [np.transpose(bn_param, axes)
                             for bn_param in bn_parameters]

        if ip_func.info.type_name == 'Affine':
            bn_parameters = [bn_param.reshape(
                bn_param.shape[1], bn_param.shape[0]) for bn_param in bn_parameters]

        return bn_parameters

    def _compute_folded_parameters(self, ip_func, bn_func):
        beta_data = bn_func.inputs[1].d.copy()
        gamma_data = bn_func.inputs[2].d.copy()
        mean_data = bn_func.inputs[3].d.copy()
        var_data = bn_func.inputs[4].d.copy()
        eps_data = bn_func.info.args['eps']

        # Reshape BN parameters to make it match with the weight of Conv/Deconv/Affine
        beta_data, gamma_data, mean_data, var_data = self.reshape_bn_parameters(
            ip_func, [beta_data, gamma_data, mean_data, var_data], self._channel_last)
        std_data = np.sqrt(var_data + eps_data)

        # Fold
        c0 = gamma_data / std_data
        c1 = beta_data - (gamma_data * mean_data) / std_data
        w = ip_func.inputs[1]
        w_data = w.d
        w_data = c0 * w_data
        b_data = c1

        if len(ip_func.inputs) == 3:
            b = ip_func.inputs[2]
            b_data += c0 * b.d.reshape(b_data.shape)

        return w_data, np.squeeze(b_data)


class BatchNormalizationFoldingOppositeModifierInner(FunctionModifier, BatchNormBase):
    """
    Single `BatchNormalization -> Convolution` pass is folded into one `Convolution`.

    If there is a `BatchNormalization -> Convolution` pass,
    fold the batch normalization parameters to the kernel
    and bias (if it exists) of the preceding convolution,
    then skip the batch normalization following the convolution.

    Supported folding functions: `Convolution`, `Deconvolution`, `Affine`.

    """

    def __init__(self, channel_last=False):
        super(BatchNormalizationFoldingOppositeModifierInner, self).__init__()
        self._channel_last = channel_last

    def modify(self, f, inputs):
        outputs = f.outputs[0]

        # Not end
        if len(outputs.function_references) == 0:
            return

        # If not BatchNorm and previous func is not BatchNorm
        prev_func = f.inputs[0].parent
        if f.info.type_name != 'BatchNormalization':
            if prev_func == None \
               or prev_func.info.type_name != 'BatchNormalization':
                return

        # Function is BN whose next function is an inner-prod layer
        if f.info.type_name == 'BatchNormalization':
            for fr in outputs.function_references:
                if fr.info.type_name in self._fct_set:
                    return inputs[0]

        # Variable is not forked
        if len(f.outputs[0].function_references) != 1:
            return

        # Function is inner-product layer
        if not f.info.type_name in self._fct_set:
            return

        ip_func = f
        bn_func = prev_func
        w_data, b_data = self._compute_folded_parameters(ip_func, bn_func)
        ip_func.inputs[1].d = w_data
        ip_func.inputs[2].d = b_data

        x = inputs[0]
        w = ip_func.inputs[1]
        b = ip_func.inputs[2]

        h = self.connect(f, x, w, b)

        return h

    def reshape_bn_parameters(self, ip_func, bn_parameters, channel_last):
        """
                                Weight layout
        conv(channel_first)     o,i,k1,k2,...kn
        conv(channel_last)      o,k1,k2,...,kn,i
        deconv(channel_first)   i,o,k1,k2,...,kn
        deconv(channel_last)    i,k1,k2,...,kn,o
        """
        if ip_func.info.type_name == 'Deconvolution':
            axes = list(range(bn_parameters[0].ndim))
            axis_to_switch = -1 if channel_last else 1
            axes[0], axes[axis_to_switch] = axes[axis_to_switch], axes[0]
            bn_parameters = [np.transpose(bn_param, axes)
                             for bn_param in bn_parameters]

        if ip_func.info.type_name == 'Affine':
            bn_parameters = [bn_param.reshape(
                bn_param.shape[1], bn_param.shape[0], 1, 1) for bn_param in bn_parameters]

        return bn_parameters

    def _compute_folded_parameters(self, ip_func, bn_func):
        beta_data = bn_func.inputs[1].d.copy()
        gamma_data = bn_func.inputs[2].d.copy()
        mean_data = bn_func.inputs[3].d.copy()
        var_data = bn_func.inputs[4].d.copy()
        eps_data = bn_func.info.args['eps']

        # Reshape BN parameters to make it match with the weight of Conv/Deconv/Affine
        beta_data, gamma_data, mean_data, var_data = self.reshape_bn_parameters(ip_func,
                                                                                [beta_data, gamma_data, mean_data,
                                                                                 var_data], self._channel_last)
        std_data = np.sqrt(var_data + eps_data)

        c0 = gamma_data / std_data
        c1 = beta_data - (gamma_data * mean_data) / std_data
        w = ip_func.inputs[1]
        w_data = w.d

        # Reshape the weight of Affine
        if ip_func.info.type_name == 'Affine':
            d_0, d_1, d_2 = bn_func.inputs[0].shape[1:]
            d_3 = w_data.shape[1:]
            w_data = np.reshape(w_data, (d_0, d_1, d_2, d_3))

        # Fold
        w_data = c0 * w_data
        b_data = w_data * c1

        w_data = np.reshape(
            w_data, (-1, d_3)) if ip_func.info.type_name == 'Affine' else w_data

        # Reduce the dimension of bias
        # Default setting for Conv
        axes_to_reduce = tuple(range(1, w_data.ndim))
        if ip_func.info.type_name == 'Deconvolution':
            axes_to_reduce = tuple(range(
                w_data.ndim-1)) if self._channel_last else (0,) + tuple(range(2, w_data.ndim))
        axes_to_reduce = (
            0, 1, 2) if ip_func.info.type_name == 'Affine' else axes_to_reduce
        b_data = np.sum(b_data, axes_to_reduce)

        if len(ip_func.inputs) == 3:
            b = ip_func.inputs[2]
            b_data += b.d
        return w_data, b_data


class BatchNormalizationFoldingModifier(object):
    """
    Single `Convolution -> BatchNormalization` pass is folded into one `Convolution`.

    If there is a `Convolution -> BatchNormalization` pass,
    fold the batch normalization parameters to the kernel
    and bias (if it exists) of the preceding convolution,
    then skip the batch normalization following the convolution.

    Supported folding functions: `Convolution`, `Deconvolution`, `Affine`.

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.BatchNormalizationFoldingModifier()]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)

    """

    def __new__(self, opposite=False, channel_last=False):
        modifiers = [AddBiasModifier()]
        if not opposite:
            modifiers.append(BatchNormalizationFoldingModifierInner(
                channel_last=channel_last))
        else:
            modifiers.append(BatchNormalizationFoldingOppositeModifierInner(
                channel_last=channel_last))
        return modifiers
