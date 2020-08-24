# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
        if not f.info.type_name in self._fct_set:
            return

        x = inputs[0]
        w = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None

        if b is not None:
            return

        scope = self.get_parameter_scope(w)
        n_outmaps = w.shape[0]
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
        ip_func.inputs[1].d = w_data
        ip_func.inputs[2].d = b_data

        x = inputs[0]
        w = ip_func.inputs[1]
        b = ip_func.inputs[2]

        h = self.connect(f, x, w, b)

        return h

    def _compute_folded_parameters(self, ip_func, bn_func):
        # Squeeze
        beta_data = np.squeeze(bn_func.inputs[1].d)
        gamma_data = np.squeeze(bn_func.inputs[2].d)
        mean_data = np.squeeze(bn_func.inputs[3].d)
        var_data = np.squeeze(bn_func.inputs[4].d)
        eps_data = bn_func.info.args['eps']

        # Reshape
        w = ip_func.inputs[1]
        r_shape = [1 for _ in range(len(w.shape) - len(beta_data.shape))]
        beta_data = beta_data.reshape(list(beta_data.shape) + r_shape)
        gamma_data = gamma_data.reshape(list(gamma_data.shape) + r_shape)
        mean_data = mean_data.reshape(list(mean_data.shape) + r_shape)
        var_data = var_data.reshape(list(var_data.shape) + r_shape)
        sigma_data = np.sqrt(var_data + eps_data)

        # Reshape again if affine
        if ip_func.name == 'Affine':  # (inp, out) -> (out, inp)
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

    def _compute_folded_parameters(self, ip_func, bn_func):
        # Squeeze
        beta_data = np.squeeze(bn_func.inputs[1].d)
        gamma_data = np.squeeze(bn_func.inputs[2].d)
        mean_data = np.squeeze(bn_func.inputs[3].d)
        var_data = np.squeeze(bn_func.inputs[4].d)
        eps_data = bn_func.info.args['eps']

        # Reshape
        w = ip_func.inputs[1]
        r_shape = [1 for _ in range(len(w.shape) - len(beta_data.shape))]
        if self._channel_last:
            beta_data = beta_data.reshape(r_shape + list(beta_data.shape))
            gamma_data = gamma_data.reshape(r_shape + list(gamma_data.shape))
            mean_data = mean_data.reshape(r_shape + list(mean_data.shape))
            var_data = var_data.reshape(r_shape + list(var_data.shape))
        else:
            beta_data = beta_data.reshape(
                [r_shape[0]] + list(beta_data.shape) + r_shape[1:len(r_shape)])
            gamma_data = gamma_data.reshape(
                [r_shape[0]] + list(gamma_data.shape) + r_shape[1:len(r_shape)])
            mean_data = mean_data.reshape(
                [r_shape[0]] + list(mean_data.shape) + r_shape[1:len(r_shape)])
            var_data = var_data.reshape(
                [r_shape[0]] + list(var_data.shape) + r_shape[1:len(r_shape)])

        sigma_data = np.sqrt(var_data + eps_data)

        # Reshape again if affine
        if ip_func.name == 'Affine':  # (inp, out) -> (out, inp)
            beta_data = beta_data.reshape(
                beta_data.shape[1], beta_data.shape[0], 1, 1)
            gamma_data = gamma_data.reshape(
                gamma_data.shape[1], gamma_data.shape[0], 1, 1)
            mean_data = mean_data.reshape(
                mean_data.shape[1], mean_data.shape[0], 1, 1)
            var_data = var_data.reshape(
                var_data.shape[1], var_data.shape[0], 1, 1)
            sigma_data = np.sqrt(var_data + eps_data)

        # Fold
        c0 = gamma_data / sigma_data
        c1 = beta_data - (gamma_data * mean_data) / sigma_data
        w_data = w.d

        if ip_func.name == 'Affine':
            _, d_0, d_1, d_2 = bn_func.inputs[0].shape
            _, d_3 = w_data.shape
            w_data = np.reshape(w_data, (d_0, d_1, d_2, d_3))
            w_data = c0 * w_data
            b_data = w_data * c1
            b_data = np.sum(w_data, (0, 1, 2))
            w_data = np.reshape(w_data, (-1, d_3))
        else:
            w_data = c0 * w_data
            b_data = w_data * c1
            b_data = np.sum(w_data, (1, 2, 3))

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
