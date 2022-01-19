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

from .graph_converter import FunctionModifier


class ChannelLastModifier(FunctionModifier):
    """
    Convert graph shape from Channel first (NCHW) to Channel last (NHWC) format.

    Supported functions: `Convolution`, `Deconvolution`, `BatchNormalization`,
    `MaxPooling`, `AveragePooling`, `SumPooling`, `Unpooling`, `Concatenate`

    Args:
        inputs (list of nn.Variable): Original very begining inputs (NCHW) of a network.
        inputs_cl (list of nn.Variable): Channel last version of very begining inputs (NHWC) of a network.
          If this is not given, `inputs_cl` are generated internally and holded.

    Examples:

    .. code-block:: python

       pred = Model(...)

       import nnabla.experimental.graph_converters as GC

       modifiers = [GC.ChannelLastModifier(<inputs of pred>)]
       gc = GC.GraphConverter(modifiers)
       pred = gc.convert(pred)

    """

    def __init__(self, inputs, inputs_cl=None):
        super(ChannelLastModifier, self).__init__()

        self._inputs = inputs
        self._inputs_cl = inputs_cl

        self._prepare_inputs(inputs, inputs_cl)

    def _prepare_inputs(self, inputs, inputs_cl=None):
        if inputs_cl is None:
            inputs_cl = []
            for inp in inputs:
                b, c, h, w = inp.shape
                x = nn.Variable([b, h, w, c])
                x.d = inp.d.copy().transpose([0, 2, 3, 1])
                inputs_cl.append(x)
        self.inputs_cl = inputs_cl

        # Replace the very begining of input
        for inp, inp_cl in zip(inputs, inputs_cl):
            f = inp.function_references[0]
            self.init_map_func_inputs(f, [inp_cl])

    def connect(self, fname, inputs, args):
        if fname in ['Convolution', 'Deconvolution']:
            # TODO: address leading batch dimension
            args['channel_last'] = True
            x = inputs[0]
            w = inputs[1]
            b = inputs[2] if len(inputs) == 3 else None
            scope = self.get_parameter_scope(w)
            with nn.parameter_scope(scope):
                wd = w.d.copy().transpose(0, 2, 3, 1)
                w = nn.parameter.get_parameter_or_create('W_cl', wd.shape, wd)
            o = F.convolution(x, w, b, **args)
        elif fname == 'BatchNormalization':
            # TODO: address leading batch dimension
            x = inputs[0]
            beta = inputs[1]
            gamma = inputs[2]
            mean = inputs[3]
            var = inputs[4]
            args['axes'] = [len(x.shape) - 1]
            if 'no_scale' in args:
                del args['no_scale']
            if 'no_bias' in args:
                del args['no_bias']
            scope = self.get_parameter_scope(beta)
            with nn.parameter_scope(scope):
                beta_d = beta.d.copy().transpose(0, 2, 3, 1)
                gamma_d = gamma.d.copy().transpose(0, 2, 3, 1)
                mean_d = mean.d.copy().transpose(0, 2, 3, 1)
                var_d = var.d.copy().transpose(0, 2, 3, 1)
                beta = nn.parameter.get_parameter_or_create(
                    'beta_cl', beta_d.shape, beta_d, beta.need_grad)
                gamma = nn.parameter.get_parameter_or_create(
                    'gamma_cl', gamma_d.shape, gamma_d, gamma.need_grad)
                mean = nn.parameter.get_parameter_or_create(
                    'mean_cl', mean_d.shape, mean_d, mean.need_grad)
                var = nn.parameter.get_parameter_or_create(
                    'var_cl', var_d.shape, var_d, var.need_grad)
            o = F.batch_normalization(x, beta, gamma, mean, var, **args)
        elif fname in ['MaxPooling', 'AveragePooling', 'SumPooling']:
            args['channel_last'] = True
            o = self._call_function(fname, inputs, args)
        elif fname in ['Concatenate']:
            args['axis'] = len(inputs[0].shape) - 1
            o = self._call_function(fname, inputs, args)
        elif fname == 'Affine':
            x = inputs[0]

            _, h_s, w_s, c_s = inputs[0].shape
            _, b_s = inputs[1].shape
            wd = inputs[1].d.copy()
            wd = np.reshape(wd, (c_s, h_s, w_s, b_s))
            wd = np.transpose(wd, (1, 2, 0, 3))
            wd = np.reshape(wd, (-1, b_s))
            w = nn.parameter.get_parameter_or_create(
                'w_cl', wd.shape, wd, False)

            b = inputs[2] if len(inputs) == 3 else None
            o = F.affine(x, w, b, **args)
        else:
            o = self._call_function(fname, inputs, args)
        return o

    def modify(self, f, inputs):
        fname = f.info.type_name
        args = f.info.args
        if fname in ['Convolution', 'Deconvolution',
                     'BatchNormalization',
                     'MaxPooling', 'AveragePooling', 'SumPooling', 'Unpooling',
                     'Concatenate', 'Affine']:
            o = self.connect(fname, inputs, args)
            return o

    def __finish__(self):
        self._prepare_inputs(self._inputs, self._inputs_cl)
