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
from collections import defaultdict

from .graph_converter import FunctionModifier
from .batch_normalization_folding import BatchNormalizationFoldingModifier
from .batch_normalization_self_folding import BatchNormalizationSelfFoldingModifier
from .remove_function import RemoveFunctionModifier


class QuantizeNonQNNToRecordingModifier(FunctionModifier):
    def __init__(self, functions_ranks, config=None, training=True):
        super(QuantizeNonQNNToRecordingModifier, self).__init__()
        self._config = config
        self._fct_bin_set = {
            'Add2': F.add2,
            'Sub2': F.sub2,
            'Mul2': F.mul2,
            'Div2': F.div2,
            'Pow2': F.pow2
        }
        self._training = training

        # Dict to record the rank of each function
        self.functions_ranks = functions_ranks

    def get_function_rank(self, f):
        rank = self.functions_ranks.get(f, -1)
        return rank

    def check(self, f):
        def backward_traverse(f, l):
            # list functions between input and f
            for inp in f.inputs:
                if inp.parent is not None:
                    l.append(inp.parent)
                    backward_traverse(inp.parent, l)

        def forward_traverse(f, l):
            # list functions between f and output
            for fref in f.outputs[0].function_references:
                l.append(fref)
                forward_traverse(fref, l)

        def is_skip_layer(f):
            skip_inputs_layers = self._config.skip_inputs_layers
            skip_outputs_layers = self._config.skip_outputs_layers
            if not skip_inputs_layers and not skip_outputs_layers:
                return False

            fs = []
            if skip_outputs_layers:
                forward_traverse(f, fs)
            fs = list(set([func.info.type_name for func in fs]))
            is_output_layer = True if skip_outputs_layers else False
            for skl in skip_outputs_layers:
                if skl in fs:
                    is_output_layer = False
                    break

            fs = []
            if skip_inputs_layers:
                backward_traverse(f, fs)
            is_input_layer = True if skip_inputs_layers else False
            fs = list(set([func.info.type_name for func in fs]))
            for skl in skip_inputs_layers:
                if skl in fs:
                    is_input_layer = False
                    break

            for skl in skip_inputs_layers:
                if f.info.type_name == skl and is_input_layer:
                    return True
            for skl in skip_outputs_layers:
                if f.info.type_name == skl and is_output_layer:
                    return True
            return False

        fn = f.info.type_name
        cfg = self._config

        # Do not record sink and bn
        if fn == 'Sink' or fn == 'BatchNormalization':
            return False

        # Only add recorder to specific layers
        # If record_layers empty, add to all layers
        record_layers = cfg.record_layers
        if record_layers and (fn not in record_layers):
            return False

        if is_skip_layer(f):
            return False
        return True

    def share_recorder(self, f, inputs, new_inputs, cfg):
        # share quantization parameters for Add2 and Concat
        fn = f.info.type_name
        recorder_activation = cfg.recorder_activation
        recorder_weight = cfg.recorder_weight
        axes = [3] if cfg.channel_last else [1]

        if fn in ['Add2', 'Concatenate']:
            idx = 0
            min_rank = inputs[0].rank
            for i, input_var in enumerate(new_inputs[1:]):
                if input_var.rank < min_rank:
                    idx = i + 1
                    min_rank = input_var.rank
            shared_name = 'x0'
            scope = self.get_parameter_scope(new_inputs[idx].parent.inputs[1])
            for i, input_var in enumerate(new_inputs):
                if i == idx:
                    continue
                input_var = input_var.parent.inputs[0]
                with nn.parameter_scope(scope):
                    input_var = recorder_activation()(
                        input_var, axes=axes, training=self._training, name=shared_name)
                new_inputs[i] = input_var
        return new_inputs

    def add_recorder(self, f, inputs, cfg):
        fn = f.info.type_name
        function_rank = self.get_function_rank(f)
        scope = '{}-{}'.format(fn, function_rank)
        axes = [3] if cfg.channel_last else [1]
        recorder_activation = cfg.recorder_activation
        recorder_weight = cfg.recorder_weight

        params_idx = 1
        if fn in ['Concatenate', 'Stack']:
            params_idx = len(inputs)
        if fn in self._fct_bin_set:
            params_idx = 2

        new_inputs = []
        # Add recorder for variable(activation)
        for i, input_var in enumerate(inputs[:params_idx]):
            fref = input_var.function_references
            if fref and fref[0].info.type_name == cfg.recorder_activation().name():
                input_var = fref[0].outputs[0]
            else:
                with nn.parameter_scope(scope):
                    parent = input_var.parent
                    if parent and parent.info.type_name == recorder_activation().name():
                        input_var = input_var
                    else:
                        input_var = recorder_activation()(input_var, axes=axes, training=self._training,
                                                          name='x{}'.format(i))
            new_inputs.append(input_var)

        # Add recorder for parameters(weight and bias)
        for i, input_parameter in enumerate(inputs[params_idx:]):
            with nn.parameter_scope(scope):
                input_parameter = recorder_weight()(input_parameter, axes=axes, training=self._training,
                                                    name='w{}'.format(i))
            new_inputs.append(input_parameter)
        return new_inputs

    def modify(self, f, inputs):
        if not self.check(f):
            return  # Skip modify this function

        fn = f.info.type_name
        cfg = self._config
        axes = [3] if cfg.channel_last else [1]
        recorder_activation = cfg.recorder_activation

        # Add recorder for each input
        new_inputs = self.add_recorder(f, inputs, cfg)
        new_inputs = self.share_recorder(f, inputs, new_inputs, cfg)

        h = self._modify_as_same(f, new_inputs)

        # Add recorder before/after a function
        next_func = f.outputs[0].function_references[0]
        next_func_rank = self.get_function_rank(next_func)
        scope = '{}-{}'.format(next_func.info.type_name, next_func_rank)
        with nn.parameter_scope(scope):
            if cfg.recorder_position == cfg.RecorderPosition.BOTH:
                h = recorder_activation()(h, axes=axes, training=self._training, name='x0')
        return h
