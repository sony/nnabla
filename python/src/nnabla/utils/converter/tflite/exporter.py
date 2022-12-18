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


import numpy as np
import json
import os
import subprocess
from functools import partial
from enum import Enum
from nnabla.utils import nnabla_pb2
import nnabla as nn
from nnabla.core.graph_optimizer import IdentityRemover
import sys


random_seed = 0
if sys.platform == 'linux':
    fn = 'flatc_linux'
elif sys.platform == 'darwin':
    fn = 'flatc_mac'
else:
    fn = 'flatc_windows.exe'
flatc_path = os.path.join(os.path.dirname(__file__), fn)


def fork_name(name):
    global random_seed
    random_seed += 1
    ret = name + '_{:04}'.format(random_seed)
    return ret


def replace_negative_size_with_batch_size(shape, batch_size):
    """Replace all dimensions with negative values to batch size"""
    sl = []
    for d in shape:
        if d < 0:
            # Negative size means batch size
            sl.append(batch_size)
        else:
            sl.append(d)
    return sl


def generate_value(type, dims, multiplier):
    if type == 'Normal':
        ret = np.random.randn(*dims) * multiplier
    elif type == 'Uniform':
        ret = np.random.uniform(-multiplier, multiplier, size=dims)
    elif type == 'Constant':
        ret = np.ones(dims) * multiplier
    else:
        raise ValueError('Generator type "' +
                         type + '" is not supported.')
    return ret.astype(np.float32).flatten().tolist()


class DataFormat(Enum):
    channel_first = 0
    channel_last = 1
    unknown = 99


class TFLiteExporter:
    def __init__(self, nnp, batch_size, channel_last=False, data_type="float32", quantization=None, dataset=None):
        # check flatc installation
        try:
            subprocess.check_output([flatc_path, '--version'])
        except subprocess.CalledProcessError:
            raise ValueError(
                "Can't find the flatbuffers package. Please refer to the Tensorflow Lite section of this website to install flatbuffers.\n"
                + "https://nnabla.readthedocs.io/en/latest/python/file_format_converter/file_format_converter.html#overview")
        if quantization and dataset is None:
            raise ValueError("No represent dataset was provided!")
        if quantization and not os.path.exists(dataset):
            raise ValueError(
                'Represent dataset {} does not exist!'.format(dataset))
        self.nnp_proto = nnp.protobuf
        self.batch_size = batch_size
        self.variables = {}
        self.parameters = {}
        self.global_data_format = DataFormat.channel_last if channel_last else DataFormat.channel_first
        self.data_type = data_type
        self.quantization = quantization
        self.dataset = dataset
        self.models = None
        self.operator_codes_list = []
        self.operators_list = []
        self.buffer_list = ['Version']
        self.tensor_list = []
        self.tflite_model_info = {'inputs': {}, 'outputs': {}}
        self.cur_base_axis = 1
        self.resolved_parameters = []
        self.executor = None
        self.inputs = []
        self.outputs = []

        self.map_nn_func_to_tflite_op = {
            'Abs': partial(self.SignalInputGenericFunc, 'ABS'),
            'ReLU': partial(self.SignalInputGenericFunc, 'RELU'),
            'ReLU6': partial(self.SignalInputGenericFunc, 'RELU6'),
            'Sigmoid': partial(self.SignalInputGenericFunc, 'LOGISTIC'),
            'Tanh': partial(self.SignalInputGenericFunc, 'TANH'),
            'Exp': partial(self.SignalInputGenericFunc, 'EXP'),
            'Log': partial(self.SignalInputGenericFunc, 'LOG'),
            'Round': partial(self.SignalInputGenericFunc, 'ROUND'),
            'Ceil': partial(self.SignalInputGenericFunc, 'CEIL'),
            'Floor': partial(self.SignalInputGenericFunc, 'FLOOR'),
            'Sin': partial(self.SignalInputGenericFunc, 'SIN'),
            'Cos': partial(self.SignalInputGenericFunc, 'COS'),
            'Add2': partial(self.BinaryOperator, 'ADD'),
            'Sub2': partial(self.BinaryOperator, 'SUB'),
            'Mul2': partial(self.BinaryOperator, 'MUL'),
            'Div2': partial(self.BinaryOperator, 'DIV'),
            'Pow2': partial(self.BinaryOperator, 'POW'),
            'Convolution': partial(self.BaseConvolution, 'CONV_2D'),
            'DepthwiseConvolution': partial(self.BaseConvolution, 'DEPTHWISE_CONV_2D'),
            'Deconvolution': self.BaseDeconvolution,
            'DepthwiseDeconvolution': self.BaseDeconvolution,
            'MaxPooling': partial(self.BasePooling, 'MAX_POOL_2D'),
            'AveragePooling': partial(self.BasePooling, 'AVERAGE_POOL_2D'),
            'SumPooling': partial(self.BasePooling, 'AVERAGE_POOL_2D'),
            'GlobalAveragePooling': self.GlobalAveragePooling,
            'Unpooling': self.Unpooling,
            'Embed': self.Embed,
            'Swish': self.Swish,
            'ELU': self.ELU,
            'LeakyReLU': self.LeakyReLU,
            'Softmax': self.Softmax,
            'LogSoftmax': self.LogSoftmax,
            'PReLU': self.PReLU,
            'Reshape': self.Reshape,
            'Tan': self.Tan,
            'BatchMatmul': self.BatchMatmul,
            'Transpose': self.Transpose,
            'Concatenate': self.Concatenate,
            'Split': self.Split,
            'Stack': self.Stack,
            'Slice': self.Slice,
            'Pad': self.Pad,
            'Tile': self.Tile,
            'Flip': self.Flip,
            'Broadcast': self.Broadcast,
            'MulScalar': partial(self.ScalarOperator, 'MUL'),
            'AddScalar': partial(self.ScalarOperator, 'ADD'),
            'PowScalar': partial(self.ScalarOperator, 'POW'),
            'RSubScalar': partial(self.ScalarOperator, 'SUB'),
            'RDivScalar': partial(self.ScalarOperator, 'DIV'),
            'RPowScalar': partial(self.ScalarOperator, 'POW'),
            'Sum': partial(self.ReductionOperator, 'SUM'),
            'Mean': partial(self.ReductionOperator, 'MEAN'),
            'Max': partial(self.ReductionOperator, 'REDUCE_MAX'),
            'Min': partial(self.ReductionOperator, 'REDUCE_MIN'),
            'Prod': partial(self.ReductionOperator, 'REDUCE_PROD'),
            'Minimum2': partial(self.BinaryOperator, 'MINIMUM'),
            'Maximum2': partial(self.BinaryOperator, 'MAXIMUM'),
            'MinimumScalar': partial(self.BinaryOperator, 'MINIMUM'),
            'MaximumScalar': partial(self.BinaryOperator, 'MAXIMUM'),
            'LogicalAnd': partial(self.LogicalOperator, 'LOGICAL_AND'),
            'LogicalOr': partial(self.LogicalOperator, 'LOGICAL_OR'),
            'LogicalNot': partial(self.LogicalOperator, 'LOGICAL_NOT'),
            'LogicalAndScalar': partial(self.LogicalOperator, 'LOGICAL_AND'),
            'LogicalOrScalar': partial(self.LogicalOperator, 'LOGICAL_OR'),
            'Equal': partial(self.LogicalOperator, 'EQUAL'),
            'NotEqual': partial(self.LogicalOperator, 'NOT_EQUAL'),
            'GreaterEqual': partial(self.LogicalOperator, 'GREATER_EQUAL'),
            'Greater': partial(self.LogicalOperator, 'GREATER'),
            'LessEqual': partial(self.LogicalOperator, 'LESS_EQUAL'),
            'Less': partial(self.LogicalOperator, 'LESS'),
            'EqualScalar': partial(self.LogicalOperator, 'EQUAL'),
            'NotEqualScalar': partial(self.LogicalOperator, 'NOT_EQUAL'),
            'GreaterEqualScalar': partial(self.LogicalOperator, 'GREATER_EQUAL'),
            'GreaterScalar': partial(self.LogicalOperator, 'GREATER'),
            'LessEqualScalar': partial(self.LogicalOperator, 'LESS_EQUAL'),
            'LessScalar': partial(self.LogicalOperator, 'LESS'),
            'LogicalXor': partial(self.LogicalOperator, None),
            'LogicalXorScalar': partial(self.LogicalOperator, None),
            'BatchNormalization': self.BatchNormalization,
            'FusedBatchNormalization': self.FusedBatchNormalization,
            'Affine': self.Affine,
            'Interpolate': self.Interpolate,
        }

    def create_parameter(self, name, shape, data):
        p = nnabla_pb2.Parameter()
        p.variable_name = name
        p.shape.dim.extend(shape)
        p.data.extend(data)
        self.parameters[name] = p
        self.variables[name] = shape

    def generate_reshape_op(self, input, output, shape):
        shape_name = fork_name(input) + "_shape"
        self.create_parameter(shape_name, [len(shape)], shape)
        self.convert_generic_tflite_op(
            [input, shape_name], [output], "RESHAPE", tensor_type={shape_name: 'INT32'})

    def generate_pad_op(self, input, output, padding):
        padding_name = fork_name(input) + "_padding"
        self.create_parameter(
            padding_name, [len(self.variables[input]), 2], padding.flatten().tolist())

        tensor_type = {padding_name: 'INT32'}
        self.convert_generic_tflite_op([input, padding_name], [
                                       output], "PAD", tensor_type=tensor_type)

    def generate_split_op(self, input, outputs, axis, num_splits):
        axis_name = fork_name(input) + "_axis"
        self.create_parameter(axis_name, [1], [axis])

        tensor_type = {axis_name: 'INT32'}
        self.convert_generic_tflite_op(
            [axis_name, input], outputs, "SPLIT", tensor_type=tensor_type)
        self.operators_list[-1]['builtin_options_type'] = 'SplitOptions'
        self.operators_list[-1]['builtin_options'] = {'num_splits': num_splits}

    def generate_bn_op(self, inputs, outputs, eps, axes, batch_stat):
        input_shape = self.variables[inputs[0]]
        reduc_axes = list(range(len(input_shape)))
        del reduc_axes[axes]
        if batch_stat:
            axes_name = fork_name(inputs[0]) + "_axes"
            self.create_parameter(axes_name, [len(reduc_axes)], reduc_axes)

            reduce_size = fork_name(inputs[0]) + "_reduce_size"
            self.create_parameter(
                reduce_size, [1], [np.prod(input_shape) / input_shape[axes]])

            tensor_type = {axes_name: 'INT32'}
            mean_out = fork_name(inputs[0]) + "_mean"
            self.variables[mean_out] = [input_shape[i] if i ==
                                        axes else 1 for i in range(len(input_shape))]
            self.convert_generic_tflite_op(
                [inputs[0], axes_name], [mean_out], 'MEAN', tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
            self.operators_list[-1]['builtin_options'] = {'keep_dims': True}
            inputs[3] = mean_out

            sub_out = fork_name(inputs[0]) + "_sub"
            self.variables[sub_out] = input_shape
            self.convert_generic_tflite_op(
                [inputs[0], mean_out], [sub_out], 'SUB')

            mul_out = fork_name(inputs[0]) + "_mul"
            self.variables[mul_out] = input_shape
            self.convert_generic_tflite_op(
                [sub_out, sub_out], [mul_out], 'MUL')

            sum_out = fork_name(inputs[0]) + "_sum"
            self.variables[sum_out] = [input_shape[i] if i ==
                                       axes else 1 for i in range(len(input_shape))]
            self.convert_generic_tflite_op(
                [mul_out, axes_name], [sum_out], 'SUM', tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
            self.operators_list[-1]['builtin_options'] = {'keep_dims': True}

            div_out = fork_name(inputs[0]) + "_div"
            self.variables[div_out] = [input_shape[i] if i ==
                                       axes else 1 for i in range(len(input_shape))]
            self.convert_generic_tflite_op(
                [sum_out, reduce_size], [div_out], 'DIV')
            inputs[4] = div_out

        if self.quantization:
            mean = self.parameters[inputs[3]]
            var = self.parameters[inputs[4]]
            beta = self.parameters[inputs[1]]
            gamma = self.parameters[inputs[2]]
            var_data = np.array(list(var.data), dtype=np.float32)
            std_data = np.sqrt(var_data + eps)
            gamma_data = np.array(list(gamma.data), dtype=np.float32)
            _gamma_data = gamma_data / std_data
            # clean data
            for _ in range(len(gamma.data)):
                gamma.data.pop()
            gamma.data.extend(_gamma_data.tolist())
            beta_data = np.array(list(beta.data), dtype=np.float32)
            mean_data = np.array(list(mean.data), dtype=np.float32)
            _beta_data = -(gamma_data * mean_data / std_data) + beta_data
            # clean data
            for _ in range(len(beta.data)):
                beta.data.pop()
            beta.data.extend(_beta_data.tolist())
            mul_out = fork_name(inputs[0]) + "_mul"
            self.variables[mul_out] = input_shape
            self.convert_generic_tflite_op(
                [inputs[0], inputs[2]], [mul_out], 'MUL')
            self.convert_generic_tflite_op(
                [mul_out, inputs[1]], outputs, 'ADD')
        else:
            sub_out = fork_name(inputs[0]) + "_sub"
            self.variables[sub_out] = input_shape
            self.convert_generic_tflite_op(
                [inputs[0], inputs[3]], [sub_out], 'SUB')

            eps_name = fork_name(inputs[0]) + "_eps"
            self.create_parameter(eps_name, [1], [eps])

            add_out = fork_name(inputs[0]) + "_sub"
            self.variables[add_out] = self.variables[inputs[4]]
            self.convert_generic_tflite_op(
                [inputs[4], eps_name], [add_out], 'ADD')

            sqrt_out = fork_name(inputs[0]) + "_sqrt"
            self.variables[sqrt_out] = self.variables[inputs[4]]
            self.convert_generic_tflite_op([add_out], [sqrt_out], 'SQRT')

            div_out = fork_name(inputs[0]) + "_div"
            self.variables[div_out] = input_shape
            self.convert_generic_tflite_op(
                [sub_out, sqrt_out], [div_out], 'DIV')

            mul_out = fork_name(inputs[0]) + "_mul"
            self.variables[mul_out] = input_shape
            self.convert_generic_tflite_op(
                [div_out, inputs[2]], [mul_out], 'MUL')
            self.convert_generic_tflite_op(
                [mul_out, inputs[1]], outputs, 'ADD')

    def format_nn_function(self, input, output, to_2d=True):
        input_shape = self.variables[input][:]
        output_shape = self.variables[output][:]
        if self.cur_base_axis > 1:
            input_shape = [int(np.prod(input_shape[:self.cur_base_axis]))
                           ] + input_shape[self.cur_base_axis:]
            output_shape = [
                int(np.prod(output_shape[:self.cur_base_axis]))] + output_shape[self.cur_base_axis:]
        if to_2d:
            input_shape.insert(-1, 1)
            output_shape.insert(-1, 1)
        if input_shape != self.variables[input]:
            before_reshape = fork_name(input) + "_reshape"
            after_reshape = fork_name(output) + "_reshape"
            self.variables[before_reshape] = input_shape
            self.variables[after_reshape] = output_shape
            self.generate_reshape_op(
                input, before_reshape, input_shape)
            self.generate_reshape_op(
                after_reshape, output, self.variables[output])
            return before_reshape, after_reshape
        else:
            return input, output

    def propagate_variable_semantic(self, pf):
        data_format = self.network.variables[pf.inputs[0]].data_format
        self.network.variables[pf.outputs[0]].data_format = data_format
        self.variables[pf.outputs[0]] = self.variables[pf.inputs[0]]

    def convert_tensor_data_format_to_channel_last(self, input):
        # convert channel_first to channel_last
        input_shape = self.variables[input]
        axes = list(range(len(input_shape)))
        axes.append(axes.pop(self.cur_base_axis))
        axes_name = fork_name(input) + "_axes"
        self.create_parameter(axes_name, [len(axes)], axes)
        tensor_type = {axes_name: 'INT32'}
        new_input_shape = input_shape.copy()
        new_input_shape.append(new_input_shape.pop(self.cur_base_axis))
        transpose_out = fork_name(input) + '_transpose'
        self.variables[transpose_out] = new_input_shape
        self.convert_generic_tflite_op(
            [input, axes_name], [transpose_out], "TRANSPOSE", tensor_type=tensor_type)
        return transpose_out

    def convert_tensor_data_format_to_channel_first(self, input):
        # convert channel_last to channel_first
        input_shape = self.variables[input]
        axes = list(range(len(input_shape)))
        axes.insert(self.cur_base_axis, axes.pop(-1))
        axes_name = fork_name(input) + "_axes"
        self.create_parameter(axes_name, [len(axes)], axes)
        tensor_type = {axes_name: 'INT32'}
        new_input_shape = input_shape.copy()
        new_input_shape.insert(self.cur_base_axis, new_input_shape.pop(-1))
        transpose_out = fork_name(input) + '_transpose'
        self.variables[transpose_out] = new_input_shape
        self.convert_generic_tflite_op(
            [input, axes_name], [transpose_out], "TRANSPOSE", tensor_type=tensor_type)
        return transpose_out

    def convert_parameter_data_format(self, pv, is_transpose_conv=False):
        weight = self.parameters[pv]
        shape = list(weight.shape.dim)
        if len(shape) >= 3 and pv not in self.resolved_parameters:
            var_d = np.array(weight.data, dtype=np.float32).reshape(shape)
            axes_order = list(range(len(shape)))
            if is_transpose_conv:
                shape.append(shape.pop(0))
                axes_order.append(axes_order.pop(0))
            else:
                shape.append(shape.pop(1))
                axes_order.append(axes_order.pop(1))
            var_d = np.transpose(var_d, axes_order).flatten()
            self.create_parameter(pv, shape, var_d)
            self.resolved_parameters.append(pv)

    def check_resolve_axis_attr(self, pf):
        axis = pf.args['axis']
        all_data_format = []
        for inp in pf.inputs:
            if inp in self.parameters:
                all_data_format.append(self.global_data_format)
            else:
                all_data_format.append(self.network.variables[inp].data_format)
        if len(set(all_data_format)) > 1:
            outs = []
            if axis == self.cur_base_axis:
                axis = len(self.variables[pf.inputs[0]]) - 1
            else:
                axis = axis - 1 if axis > self.cur_base_axis else axis
            for i, inp in enumerate(pf.inputs):
                if all_data_format[i] == DataFormat.channel_first:
                    if inp in self.inputs:
                        outs.append(inp)
                        self.variables[inp].append(
                            self.variables[inp].pop(self.cur_base_axis))
                        self.network.variables[inp].data_format = DataFormat.channel_last
                        self.tflite_model_info['inputs'][inp] = {
                            'transpose': True, 'base_axis': self.cur_base_axis}
                    elif inp in self.parameters:
                        self.convert_parameter_data_format(inp)
                        outs.append(inp)
                    else:
                        o = self.convert_tensor_data_format_to_channel_last(
                            inp)
                        outs.append(o)
                else:
                    outs.append(inp)
            for outp in pf.outputs:
                self.network.variables[outp].data_format = DataFormat.channel_last
            return True, outs, axis
        else:
            data_format = self.network.variables[pf.inputs[0]].data_format
            input_shape = self.variables[pf.inputs[0]]
            if data_format == DataFormat.channel_last and \
                    self.global_data_format == DataFormat.channel_first:
                if axis == self.cur_base_axis:
                    axis = len(input_shape) - 1
                else:
                    axis = axis - 1 if axis > self.cur_base_axis else axis
                for outp in pf.outputs:
                    self.network.variables[outp].data_format = DataFormat.channel_last
                return True, pf.inputs, axis
            else:
                for outp in pf.outputs:
                    self.network.variables[outp].data_format = data_format
                return False, pf.inputs, axis

    def check_resolve_broadcast(self, inputs, outputs):
        data_format0 = self.network.variables[inputs[0]
                                              ].data_format if inputs[0] in self.network.variables else self.global_data_format
        data_format1 = self.network.variables[inputs[1]
                                              ].data_format if inputs[1] in self.network.variables else self.global_data_format
        resolve_inputs = inputs.copy()
        shape_a = self.variables[resolve_inputs[0]]
        shape_b = self.variables[resolve_inputs[1]]
        if data_format0 != data_format1:
            if len(shape_a) < len(shape_b):
                diff = len(shape_b) - len(shape_a)
                new_shape_a = [1] * diff + shape_a
                rout = fork_name(resolve_inputs[0]) + "_reshpae"
                self.variables[rout] = new_shape_a
                self.generate_reshape_op(resolve_inputs[0], rout, new_shape_a)
                resolve_inputs[0] = rout
            elif len(shape_a) > len(shape_b):
                diff = len(shape_a) - len(shape_b)
                new_shape_b = [1] * diff + shape_b
                rout = fork_name(resolve_inputs[1]) + "_reshpae"
                self.variables[rout] = new_shape_b
                self.generate_reshape_op(resolve_inputs[1], rout, new_shape_b)
                resolve_inputs[1] = rout

            if data_format0 == DataFormat.channel_first:
                resolve_inputs[0] = self.convert_tensor_data_format_to_channel_last(
                    resolve_inputs[0])
            if data_format1 == DataFormat.channel_first:
                resolve_inputs[1] = self.convert_tensor_data_format_to_channel_last(
                    resolve_inputs[1])
            new_shape_a = self.variables[resolve_inputs[0]]
            new_shape_b = self.variables[resolve_inputs[1]]
            output_shape = [new_shape_b[i] if new_shape_a[i] ==
                            1 else new_shape_a[i] for i in range(len(new_shape_a))]
            self.variables[outputs[0]] = output_shape
            self.network.variables[outputs[0]
                                   ].data_format = DataFormat.channel_last
        else:
            dims = len(shape_a) if len(shape_a) > len(
                shape_b) else len(shape_b)
            new_shape_a = [1] * (dims - len(shape_a)) + shape_a
            new_shape_b = [1] * (dims - len(shape_b)) + shape_b
            self.variables[outputs[0]] = [
                new_shape_b[i] if new_shape_a[i] == 1 else new_shape_a[i] for i in range(dims)]
            self.network.variables[outputs[0]].data_format = data_format0
        return resolve_inputs

    def check_resolve_input_semantic_divergence(self, inputs, outputs, is_transpose_conv=False):
        # Check for semantic divergence. if it exists:
        # 1. resove the input tensor by adding transpose.
        # 2. parameter will be directly transposed and saved.
        # 3. modify outputs shape and set data_format to outputs variables
        resolve_inputs = []
        if self.global_data_format == DataFormat.channel_first:
            for inp in inputs[:]:
                if inp in self.parameters:
                    # convert weight data_format
                    self.convert_parameter_data_format(
                        inp, is_transpose_conv=is_transpose_conv)
                    resolve_inputs.append(inp)
                else:
                    # convert input tensor data_format
                    data_format = self.network.variables[inp].data_format
                    if data_format == DataFormat.channel_first:
                        new_inp = self.convert_tensor_data_format_to_channel_last(
                            inp)
                        resolve_inputs.append(new_inp)
                    else:
                        resolve_inputs.append(inp)
            for outp in outputs:
                # modify output shape and data_format
                output_shape = self.variables[outp].copy()
                output_shape.append(output_shape.pop(self.cur_base_axis))
                self.variables[outp] = output_shape
                self.network.variables[outp].data_format = DataFormat.channel_last
        else:
            for outp in outputs:
                self.network.variables[outp].data_format = DataFormat.channel_last
        return resolve_inputs

    def operators_sorting(self, pf):
        op_num = len(self.operators_list)
        if op_num == 1:
            self.models['subgraphs'][0]['operators'].append(
                self.operators_list[0])
        else:
            from collections import Counter
            cnt = Counter()
            for op in self.operators_list:
                for index in op['inputs']:
                    name = self.tensor_list[index]
                    cnt[name] += 1

            inputs = pf.inputs[:]
            for _ in range(op_num):
                for index, op in enumerate(self.operators_list[:]):
                    op_inputs = [self.tensor_list[i] for i in op['inputs']
                                 if self.tensor_list[i] not in self.parameters]
                    op_outputs = [self.tensor_list[i] for i in op['outputs']]
                    if set(op_inputs) <= set(inputs):
                        for inp in op_inputs:
                            cnt[inp] -= 1
                            if cnt[inp] == 0:
                                inputs.pop(inputs.index(inp))
                        inputs.extend(op_outputs)
                        self.models['subgraphs'][0]['operators'].append(
                            self.operators_list.pop(index))
                        break

    def SignalInputGenericFunc(self, tflite_op, pf):
        self.propagate_variable_semantic(pf)
        self.convert_generic_tflite_op(pf.inputs, pf.outputs, tflite_op)

    def BinaryOperator(self, tflite_op, pf):
        inputs = pf.inputs[:]

        if len(inputs) > 1:
            inputs = self.check_resolve_broadcast(inputs, pf.outputs)
        else:
            data_format = self.network.variables[inputs[0]].data_format
            self.network.variables[pf.outputs[0]].data_format = data_format
            self.variables[pf.outputs[0]] = self.variables[pf.inputs[0]]

        if pf.type == 'MinimumScalar' or pf.type == 'MaximumScalar':
            val = pf.args.get('val', 1.0)
            scalar_value = fork_name(pf.inputs[0]) + "_scalar"
            self.create_parameter(scalar_value, [1], [val])
            inputs.append(scalar_value)

        self.convert_generic_tflite_op(inputs, pf.outputs, tflite_op)

    def LogicalOperator(self, tflite_op, pf):
        inputs = pf.inputs[:]
        tensor_type = {}

        if len(inputs) == 2:
            inputs = self.check_resolve_broadcast(inputs, pf.outputs)
        else:
            data_format = self.network.variables[pf.inputs[0]].data_format
            if pf.type.endswith("Scalar"):
                val = pf.args.get('val', 1)
                scalar_value = fork_name(pf.inputs[0]) + "_scalar"
                self.create_parameter(scalar_value, [1], [val])
                inputs.append(scalar_value)
            self.network.variables[pf.outputs[0]].data_format = data_format
            self.variables[pf.outputs[0]] = self.variables[pf.inputs[0]]

        # set tensor_type
        tensor_type[pf.outputs[0]] = 'BOOL'
        if pf.type in ['LogicalAnd', 'LogicalOr', 'LogicalXor', 'LogicalAndScalar', 'LogicalOrScalar', 'LogicalXorScalar']:
            tensor_type[inputs[0]] = 'BOOL'
            tensor_type[inputs[1]] = 'BOOL'
        if pf.type in ['LogicalNot']:
            tensor_type[inputs[0]] = 'BOOL'

        # convert to tflite
        if pf.type == 'LogicalXor' or pf.type == 'LogicalXorScalar':
            and_out1 = fork_name(inputs[0]) + "_and"
            tensor_type[and_out1] = 'BOOL'
            self.variables[and_out1] = self.variables[inputs[0]]
            self.convert_generic_tflite_op(
                inputs, [and_out1], 'LOGICAL_AND', tensor_type=tensor_type)
            not_out = fork_name(inputs[0]) + "_not"
            tensor_type[not_out] = 'BOOL'
            self.variables[not_out] = self.variables[inputs[0]]
            self.convert_generic_tflite_op(
                [and_out1], [not_out], 'LOGICAL_NOT', tensor_type=tensor_type)
            or_out = fork_name(inputs[1]) + "_or"
            tensor_type[or_out] = 'BOOL'
            self.variables[or_out] = self.variables[inputs[1]]
            self.convert_generic_tflite_op(
                inputs, [or_out], 'LOGICAL_OR', tensor_type=tensor_type)
            self.convert_generic_tflite_op(
                [not_out, or_out], pf.outputs, 'LOGICAL_AND', tensor_type=tensor_type)
        else:
            self.convert_generic_tflite_op(
                inputs, pf.outputs, tflite_op, tensor_type=tensor_type)

    def BaseConvolution(self, tflite_op, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        weight_shape = list(self.parameters[inputs[1]].shape.dim)
        self.cur_base_axis = pf.args.get('base_axis', 1)
        builtin_options_type = 'Conv2DOptions'
        if pf.type == 'DepthwiseConvolution':
            self.parameters[inputs[1]].shape.dim.insert(0, 1)
            self.variables[inputs[1]].insert(0, 1)
            builtin_options_type = 'DepthwiseConv2DOptions'
        group = pf.args.get('group', 1)
        pads = pf.args.get('pad', [0] * (len(weight_shape) - 2))
        strides = pf.args.get('stride', [1] * (len(weight_shape) - 2))
        dilations = pf.args.get('dilation', [1] * (len(weight_shape) - 2))

        if len(weight_shape) > 4:
            raise ValueError('Currently, only supports CONV_2D')

        # add bias
        if len(inputs) == 2:
            bias_name = fork_name(inputs[0]) + "_bias"
            shape = weight_shape[0]
            data = np.zeros(shape, dtype=np.float32).flatten().tolist()
            self.create_parameter(bias_name, [shape], data)
            inputs.append(bias_name)

        inputs = self.check_resolve_input_semantic_divergence(inputs, outputs)

        if len(pads) < 2:  # 1-D convolution
            dilations += [1]
            strides += [1]
            pads += [0]
            self.parameters[inputs[1]].shape.dim.insert(-1, 1)
            self.variables[inputs[1]].insert(-1, 1)
            inputs[0], outputs[0] = self.format_nn_function(
                inputs[0], outputs[0], to_2d=True)
        else:
            inputs[0], outputs[0] = self.format_nn_function(
                inputs[0], outputs[0], to_2d=False)

        if any(pads):
            pads = [0] + pads + [0]
            pad_shape = [self.variables[inputs[0]]
                         [i] + pads[i] * 2 for i in range(len(pads))]
            padding = np.array([[p] * 2 for p in pads], dtype=np.int32)
            pad_name = fork_name(inputs[0]) + "_pad"
            self.variables[pad_name] = pad_shape
            self.generate_pad_op(inputs[0], pad_name, padding)
            inputs[0] = pad_name

        if group > 1:
            input_shape = self.variables[inputs[0]]
            new_weight_shape = self.variables[inputs[1]]
            output_shape = self.variables[outputs[0]]
            # split input
            split_i_outs = []
            for _ in range(group):
                out = fork_name(inputs[0]) + "_split"
                shape = input_shape[:-1] + [input_shape[-1] // group]
                split_i_outs.append(out)
                self.variables[out] = shape
            self.generate_split_op(inputs[0], split_i_outs, -1, group)

            # split weight
            split_w_outs = []
            weight = self.parameters[inputs[1]]
            var_d = np.array(weight.data, dtype=np.float32).reshape(
                new_weight_shape)
            split_w = np.split(var_d, group, axis=0)
            for i in range(group):
                out = fork_name(inputs[1]) + "_split_w"
                split_w_outs.append(out)
                self.variables[out] = split_w[i].shape
                self.create_parameter(
                    out, split_w[i].shape, split_w[i].flatten().tolist())

            # split bias
            split_b_outs = []
            bias = self.parameters[inputs[2]]
            var_d = np.array(bias.data, dtype=np.float32).reshape(
                (len(bias.data)))
            split_b = np.split(var_d, group, axis=0)
            for i in range(group):
                out = fork_name(inputs[2]) + "_split_b"
                split_b_outs.append(out)
                self.variables[out] = split_b[i].shape
                self.create_parameter(
                    out, split_b[i].shape, split_b[i].flatten().tolist())

            conv_outs = []
            for i in range(group):
                out = fork_name(inputs[0]) + "_conv"
                shape = output_shape[:-1] + [output_shape[-1] // group]
                conv_outs.append(out)
                self.variables[out] = shape
                self.convert_generic_tflite_op(
                    [split_i_outs[i], split_w_outs[i], split_b_outs[i]], [out], tflite_op)
                self.operators_list[-1]['builtin_options_type'] = builtin_options_type
                self.operators_list[-1]['builtin_options'] = {
                    'padding': "VALID",
                    'dilation_h_factor': dilations[0],
                    'dilation_w_factor': dilations[1],
                    'stride_h': strides[0],
                    'stride_w': strides[1],
                }
            self.convert_generic_tflite_op(
                conv_outs, outputs, "CONCATENATION")
            self.operators_list[-1]['builtin_options_type'] = 'ConcatenationOptions'
            self.operators_list[-1]['builtin_options'] = {'axis': -1}
        else:
            self.convert_generic_tflite_op(inputs, outputs, tflite_op)
            self.operators_list[-1]['builtin_options_type'] = builtin_options_type
            self.operators_list[-1]['builtin_options'] = {
                'padding': "VALID",
                'dilation_h_factor': dilations[0],
                'dilation_w_factor': dilations[1],
                'stride_h': strides[0],
                'stride_w': strides[1],
            }

    def BaseDeconvolution(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        weight_shape = list(self.parameters[inputs[1]].shape.dim)
        output_shape = self.variables[outputs[0]][:]
        self.cur_base_axis = pf.args.get('base_axis', 1)
        group = pf.args.get('group', 1)
        if pf.type == "DepthwiseDeconvolution":
            weight_shape.insert(1, 1)
            self.parameters[inputs[1]].shape.Clear()
            self.parameters[inputs[1]].shape.dim.extend(weight_shape)
            self.variables[inputs[1]] = weight_shape
            group = output_shape[self.cur_base_axis]
        pads = pf.args.get('pad', [0] * (len(weight_shape) - 2))
        strides = pf.args.get('stride', [1] * (len(weight_shape) - 2))
        dilations = pf.args.get('dilation', [1] * (len(weight_shape) - 2))
        output_padding = pf.args.get(
            'output_padding', [0] * (len(weight_shape) - 2))
        if not output_padding:
            output_padding = [0] * (len(weight_shape) - 2)

        if len(weight_shape) > 4:
            raise ValueError('Currently, only supports TRANSPOSE_CONV 2D')

        if dilations != [1] * len(dilations):
            raise ValueError(
                'Currently, TRANSPOSE_CONV does not support dilations')

        inputs = self.check_resolve_input_semantic_divergence(
            inputs, outputs, is_transpose_conv=True)

        if len(pads) < 2:  # 1-D deconvolution
            strides += [1]
            pads += [0]
            self.parameters[inputs[1]].shape.dim.insert(-1, 1)
            self.variables[inputs[1]].insert(-1, 1)
            inputs[0], outputs[0] = self.format_nn_function(
                inputs[0], outputs[0], to_2d=True)
        else:
            inputs[0], outputs[0] = self.format_nn_function(
                inputs[0], outputs[0], to_2d=False)

        # convert bias to ADD
        if len(inputs) > 2:
            add_in = fork_name(outputs[0]) + "_add"
            self.variables[add_in] = self.variables[outputs[0]]
            self.convert_generic_tflite_op([add_in, inputs[2]], outputs, "ADD")
            del inputs[2]
            outputs[0] = add_in

        new_input_shape = self.variables[inputs[0]]
        new_weight_shape = self.variables[inputs[1]]
        output_shape = []
        for index in range(2):
            i = new_input_shape[1 + index]
            w = new_weight_shape[1 + index]
            o = (i - 1) * strides[index] + w
            output_shape.append(o)
        output_shape = [new_input_shape[0]] + \
            output_shape + [new_weight_shape[0]]
        output_shape_name = fork_name(outputs[0]) + "_output_shape"
        self.create_parameter(output_shape_name, [
                              len(output_shape)], output_shape)

        # split input
        split_i_outs = []
        for _ in range(group):
            out = fork_name(inputs[0]) + "_split"
            shape = new_input_shape[:-1] + [new_input_shape[-1] // group]
            split_i_outs.append(out)
            self.variables[out] = shape
        self.generate_split_op(inputs[0], split_i_outs, -1, group)

        # split weight
        split_w_outs = []
        weight = self.parameters[inputs[1]]
        var_d = np.array(weight.data, dtype=np.float32).reshape(
            new_weight_shape)
        axes = list(range(len(new_weight_shape)))
        axes.insert(0, axes.pop(-1))
        var_d = np.transpose(var_d, axes)
        split_w = np.split(var_d, group, axis=0)
        for i in range(group):
            out = fork_name(inputs[1]) + "_split_w"
            split_w_outs.append(out)
            axes = list(range(len(new_weight_shape)))
            axes.append(axes.pop(0))
            split_w[i] = np.transpose(split_w[i], axes)
            self.variables[out] = split_w[i].shape
            self.create_parameter(
                out, split_w[i].shape, split_w[i].flatten().tolist())

        if group > 1:
            deconv_outs = []
            for i in range(group):
                deconv_out = fork_name(inputs[0]) + "_deconv"
                self.variables[deconv_out] = output_shape
                self.convert_generic_tflite_op([output_shape_name, split_w_outs[i], split_i_outs[i]], [
                                               deconv_out], "TRANSPOSE_CONV", tensor_type={output_shape_name: 'INT32'})
                self.operators_list[-1]['builtin_options_type'] = 'TransposeConvOptions'
                self.operators_list[-1]['builtin_options'] = {
                    'padding': "VALID",
                    'stride_h': strides[0],
                    'stride_w': strides[1],
                }

                # Pad
                padding = [
                    0, 0] + [j for i in zip([0, 0], output_padding) for j in i] + [0, 0]
                pad_out = fork_name(outputs[0]) + "_pad"
                self.variables[pad_out] = [self.variables[deconv_out][i] -
                                           padding[2 * i] - padding[2 * i + 1] for i in range(4)]
                padding = np.array(padding, dtype=np.int32)
                self.generate_pad_op(deconv_out, pad_out, padding)

                # Slice
                begin = [0] + pads + [0]
                begin_name = fork_name(outputs[0]) + "_slice_begin"
                size_name = fork_name(outputs[0]) + "_slice_size"
                slice_out = fork_name(outputs[0]) + "_slice"
                slice_in_shape = self.variables[pad_out]
                self.variables[slice_out] = [
                    slice_in_shape[i] - begin[i] * 2 for i in range(len(begin))]
                self.create_parameter(begin_name, [len(begin)], begin)
                self.create_parameter(
                    size_name, [len(begin)], self.variables[slice_out])
                tensor_type = {begin_name: 'INT32', size_name: 'INT32'}
                self.convert_generic_tflite_op([pad_out, begin_name, size_name], [slice_out],
                                               'SLICE', tensor_type=tensor_type)
                deconv_outs.append(slice_out)

            self.convert_generic_tflite_op(
                deconv_outs, outputs, "CONCATENATION")
            self.operators_list[-1]['builtin_options_type'] = 'ConcatenationOptions'
            self.operators_list[-1]['builtin_options'] = {'axis': -1}
        else:
            deconv_out = fork_name(inputs[0]) + "_deconv"
            self.variables[deconv_out] = output_shape
            self.convert_generic_tflite_op([output_shape_name, split_w_outs[0], split_i_outs[0]], [deconv_out],
                                           "TRANSPOSE_CONV", tensor_type={output_shape_name: 'INT32'})
            self.operators_list[-1]['builtin_options_type'] = 'TransposeConvOptions'
            self.operators_list[-1]['builtin_options'] = {
                'padding': "VALID",
                'stride_h': strides[0],
                'stride_w': strides[1],
            }

            # Pad
            padding = [
                0, 0] + [j for i in zip([0, 0], output_padding) for j in i] + [0, 0]
            pad_out = fork_name(outputs[0]) + "_pad"
            self.variables[pad_out] = [self.variables[deconv_out][i] - padding[2 * i] - padding[2 * i + 1] for i in
                                       range(4)]
            padding = np.array(padding, dtype=np.int32)
            self.generate_pad_op(deconv_out, pad_out, padding)

            # Slice
            begin = [0] + pads + [0]
            begin_name = fork_name(outputs[0]) + "_slice_begin"
            size_name = fork_name(outputs[0]) + "_slice_size"
            self.create_parameter(begin_name, [len(begin)], begin)
            self.create_parameter(
                size_name, [len(begin)], self.variables[outputs[0]])
            tensor_type = {begin_name: 'INT32', size_name: 'INT32'}
            self.convert_generic_tflite_op([pad_out, begin_name, size_name], outputs,
                                           'SLICE', tensor_type=tensor_type)

    def BasePooling(self, tflite_op, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        input_shape = self.variables[inputs[0]][:]
        value = 0.0
        k = pf.args['kernel']
        s = pf.args.get('stride', [1] * len(k))
        pads = pf.args.get('pad', [0] * len(k))
        ignore_border = pf.args.get('ignore_border', True)
        if tflite_op == 'MAX_POOL_2D' and not self.child_of_relu:
            value = 1.17e-38
        elif tflite_op == 'AVERAGE_POOL_2D':
            including_pad = pf.args.get('including_pad', True)
            if not including_pad:
                raise ValueError(
                    'Currently, disabled including_pad is not supported.')

        inputs = self.check_resolve_input_semantic_divergence(inputs, outputs)

        len_input = len(input_shape)
        len_kernel = len(k)
        if len_kernel != 2:
            raise ValueError(
                'Currently, only support 2D.')

        diff = len_input - len_kernel - 1
        if diff > 1:
            new_input_shape = self.variables[inputs[0]]
            new_output_shape = self.variables[outputs[0]]
            input_shape_reshape = np.concatenate((np.array(
                [np.prod(new_input_shape[:diff])]), np.array(new_input_shape[diff:]))).flatten().tolist()
            output_shape_reshape = np.concatenate((np.array(
                [np.prod(new_output_shape[:diff])]), np.array(new_output_shape[diff:]))).flatten().tolist()
            before_rehspae = fork_name(inputs[0]) + "_reshape"
            after_reshape = fork_name(outputs[0]) + "_reshape"
            self.variables[before_rehspae] = input_shape_reshape
            self.variables[after_reshape] = output_shape_reshape
            self.generate_reshape_op(
                inputs[0], before_rehspae, input_shape_reshape)
            self.generate_reshape_op(
                after_reshape, outputs[0], new_output_shape)
            inputs[0] = before_rehspae
            outputs[0] = after_reshape

        pads = [d for d in pads]
        if ignore_border:
            pads = [0, 0] + [j for i in zip(pads, pads) for j in i] + [0, 0]
        else:
            new_input_shape = [shape + pads[i]
                               for i, shape in enumerate(input_shape[-len_kernel:])]
            subs = [kk - i % ss if i % ss != 0 else kk - ss
                    for kk, ss, i in zip(k, s, new_input_shape)]
            pads = [0, 0] + [j for i in zip(pads, subs) for j in i] + [0, 0]

        if any(pads):
            padding = np.array(pads, dtype=np.int32)
            padding_name = fork_name(inputs[0]) + "_padding"
            self.create_parameter(
                padding_name, [2 + len_kernel, 2], padding.flatten().tolist())
            value_name = fork_name(inputs[0]) + "_value"
            self.create_parameter(value_name, [1], [value])
            tensor_type = {padding_name: 'INT32'}
            pad_out = fork_name(inputs[0]) + "_pad"
            self.variables[pad_out] = [self.variables[inputs[0]]
                                       [i] + pads[i*2] + pads[i*2+1] for i in range(2 + len_kernel)]
            if value == 0.0:
                self.generate_pad_op(inputs[0], pad_out, padding)
            else:
                self.convert_generic_tflite_op(
                    [inputs[0], padding_name, value_name], [pad_out], 'PADV2', tensor_type=tensor_type)
            inputs[0] = pad_out

        if pf.type == 'SumPooling':
            avgpool_out = fork_name(outputs[0]) + "_avg_pool"
            self.variables[avgpool_out] = self.variables[outputs[0]]
            kernel_size = int(np.prod(k))
            k_name = fork_name(inputs[0]) + "_kernel_size"
            self.create_parameter(k_name, [1], [kernel_size])
            self.convert_generic_tflite_op(
                [avgpool_out, k_name], outputs, 'MUL')
            outputs[0] = avgpool_out

        self.convert_generic_tflite_op(inputs, outputs, tflite_op)
        self.operators_list[-1]['builtin_options_type'] = 'Pool2DOptions'
        self.operators_list[-1]['builtin_options'] = {
            'padding': "VALID",
            'stride_h': s[0],
            'stride_w': s[1],
            'filter_height': k[0],
            'filter_width': k[1],
        }

    def GlobalAveragePooling(self, pf):
        input_shape = self.variables[pf.inputs[0]][:]
        spatial_size = len(input_shape) - 2
        axes = list(range(spatial_size, spatial_size + 2))
        axes_name = fork_name(pf.inputs[0]) + "_axes"
        self.create_parameter(axes_name, [len(axes)], axes)

        tensor_type = {axes_name: 'INT32'}
        self.convert_generic_tflite_op(
            [pf.inputs[0], axes_name], pf.outputs, 'MEAN', tensor_type=tensor_type)
        self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
        self.operators_list[-1]['builtin_options'] = {'keep_dims': True}
        if pf.inputs[0] not in self.network.variables:
            self.network.variables[pf.outputs[0]
                                   ].data_format = self.global_data_format
        else:
            self.propagate_variable_semantic(pf)

    def Unpooling(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        input_shape = self.variables[pf.inputs[0]][:]
        output_shape = self.variables[pf.outputs[0]][:]
        k = pf.args['kernel']
        len_k = len(k)
        len_i = len(input_shape)

        if len_k != 2:
            raise ValueError(
                'Currently, only support 2D.')

        inputs = self.check_resolve_input_semantic_divergence(inputs, outputs)

        diff = len_i - len_k - 1
        if diff > 1:
            new_input_shape = self.variables[inputs[0]]
            new_output_shape = self.variables[outputs[0]]
            input_shape_reshape = np.concatenate((np.array(
                [np.prod(new_input_shape[:diff])]), np.array(new_input_shape[diff:]))).flatten().tolist()
            output_shape_reshape = np.concatenate((np.array(
                [np.prod(new_output_shape[:diff])]), np.array(new_output_shape[diff:]))).flatten().tolist()
            before_rehspae = fork_name(inputs[0]) + "_reshape"
            after_reshape = fork_name(outputs[0]) + "_reshape"
            self.variables[before_rehspae] = input_shape_reshape
            self.variables[after_reshape] = output_shape_reshape
            self.generate_reshape_op(
                inputs[0], before_rehspae, input_shape_reshape)
            self.generate_reshape_op(
                after_reshape, outputs[0], new_output_shape)
            inputs[0] = before_rehspae
            outputs[0] = after_reshape

        # create size inputs
        size = output_shape[-len_k:]
        size_name = fork_name(inputs[0]) + "_size"
        self.create_parameter(size_name, [len(size)], size)
        tensor_type = {size_name: 'INT32'}

        self.convert_generic_tflite_op(
            [inputs[0], size_name], outputs, "RESIZE_NEAREST_NEIGHBOR", tensor_type=tensor_type)

    def Embed(self, pf):
        tensor_type = {pf.inputs[0]: 'INT32'}
        self.convert_generic_tflite_op(
            [pf.inputs[1], pf.inputs[0]], pf.outputs, 'GATHER', tensor_type=tensor_type)
        self.propagate_variable_semantic(pf)

    def Swish(self, pf):
        '''
        Convert Swish to Sigmoid(x) * x
        '''
        sigmoid_out = fork_name(pf.inputs[0]) + "_sigmoid"
        self.variables[sigmoid_out] = self.variables[pf.inputs[0]]
        self.convert_generic_tflite_op(pf.inputs, [sigmoid_out], "LOGISTIC")
        self.convert_generic_tflite_op(
            [pf.inputs[0], sigmoid_out], pf.outputs, "MUL")
        self.propagate_variable_semantic(pf)

    def ELU(self, pf):
        alpha = pf.args.get('alpha', 1.0)
        if alpha == 1.0:
            self.convert_generic_tflite_op(pf.inputs, pf.outputs, "ELU")
        else:
            raise ValueError(
                'Currently, only support alpha == 1.0')
        self.propagate_variable_semantic(pf)

    def LeakyReLU(self, pf):
        self.convert_generic_tflite_op(pf.inputs, pf.outputs, "LEAKY_RELU")
        self.operators_list[-1]['builtin_options_type'] = 'LeakyReluOptions'
        self.operators_list[-1]['builtin_options'] = {
            'alpha': pf.args.get('alpha', 0.1)}
        self.propagate_variable_semantic(pf)

    def Softmax(self, pf):
        if self.quantization:
            _, _, axis = self.check_resolve_axis_attr(pf)
            self.convert_generic_tflite_op(pf.inputs, pf.outputs, "SOFTMAX")
            self.operators_list[-1]['builtin_options_type'] = 'SoftmaxOptions'
            self.operators_list[-1]['builtin_options'] = {'beta': 1.0}
        else:
            # Convert Softmax to ReduceMax, Sub, Exp, Sum, Div
            _, _, axis = self.check_resolve_axis_attr(pf)
            axis_name = fork_name(pf.inputs[0]) + "_axis"
            self.create_parameter(axis_name, [1], [axis])
            input_shape = self.variables[pf.inputs[0]][:]
            tensor_type = {axis_name: 'INT32'}

            # ReduceMax
            mout = fork_name(pf.inputs[0]) + "_reducemax"
            mout_shape = input_shape.copy()
            mout_shape[axis] = 1
            self.variables[mout] = mout_shape
            self.convert_generic_tflite_op([pf.inputs[0], axis_name], [
                                           mout], "REDUCE_MAX", tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
            self.operators_list[-1]['builtin_options'] = {'keep_dims': True}

            # Sub
            sout = fork_name(pf.inputs[0]) + "_sub"
            self.variables[sout] = input_shape
            self.convert_generic_tflite_op([pf.inputs[0], mout], [sout], "SUB")

            # Exp
            expout = fork_name(pf.inputs[0]) + "_exp"
            self.variables[expout] = input_shape
            self.convert_generic_tflite_op([sout], [expout], "EXP")

            # Sum
            sumout = fork_name(pf.inputs[0]) + "_sum"
            sumout_shape = input_shape.copy()
            sumout_shape[axis] = 1
            self.variables[sumout] = sumout_shape
            self.convert_generic_tflite_op(
                [expout, axis_name], [sumout], "SUM", tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
            self.operators_list[-1]['builtin_options'] = {'keep_dims': True}

            # Div
            self.convert_generic_tflite_op([expout, sumout], pf.outputs, "DIV")

    def LogSoftmax(self, pf):
        # Convert LogSoftmax to ReduceMax, Sub, Exp, Sum, Log, Sub
        _, _, axis = self.check_resolve_axis_attr(pf)
        axis_name = fork_name(pf.inputs[0]) + "_axis"
        self.create_parameter(axis_name, [1], [axis])
        input_shape = self.variables[pf.inputs[0]][:]
        tensor_type = {axis_name: 'INT32'}

        # ReduceMax
        mout = fork_name(pf.inputs[0]) + "_reducemax"
        input_shape[axis] = 1
        self.variables[mout] = [1 if i == axis else input_shape[i]
                                for i in range(len(input_shape))]
        self.convert_generic_tflite_op([pf.inputs[0], axis_name], [
                                       mout], "REDUCE_MAX", tensor_type=tensor_type)
        self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
        self.operators_list[-1]['builtin_options'] = {'keep_dims': True}

        # Sub
        sout = fork_name(pf.inputs[0]) + "_sub"
        self.variables[sout] = input_shape
        self.convert_generic_tflite_op([pf.inputs[0], mout], [sout], "SUB")

        # Exp
        expout = fork_name(pf.inputs[0]) + "_exp"
        self.variables[expout] = input_shape
        self.convert_generic_tflite_op([sout], [expout], "EXP")

        # Sum
        sumout = fork_name(pf.inputs[0]) + "_sum"
        self.variables[sumout] = [1 if i == axis else input_shape[i]
                                  for i in range(len(input_shape))]
        self.convert_generic_tflite_op(
            [expout, axis_name], [sumout], "SUM", tensor_type=tensor_type)
        self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
        self.operators_list[-1]['builtin_options'] = {'keep_dims': True}

        # Log
        logout = fork_name(pf.inputs[0]) + "_log"
        self.variables[logout] = input_shape
        self.convert_generic_tflite_op([sumout], [logout], "LOG")

        # Sub
        self.convert_generic_tflite_op([sout, logout], pf.outputs, "SUB")

    def PReLU(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        self.cur_base_axis = pf.args.get('base_axis', 1)
        slope_shape = self.variables[pf.inputs[1]][:]

        if len(slope_shape) == 0:
            slope_shape.append(1)
        if len(slope_shape) != 1:
            raise ValueError("The negative slope must be a 1d")

        inputs[0], outputs[0] = self.format_nn_function(
            inputs[0], outputs[0], to_2d=False)

        self.convert_generic_tflite_op(inputs, outputs, 'PRELU')
        self.propagate_variable_semantic(pf)

    def Tan(self, pf):
        '''
        Convert Tan to Sin(x) / Cos(x)
        '''
        sin_out = fork_name(pf.inputs[0]) + "_sin"
        cos_out = fork_name(pf.inputs[0]) + "_cos"
        self.variables[sin_out] = self.variables[cos_out] = self.variables[pf.inputs[0]]
        self.convert_generic_tflite_op([pf.inputs[0]], [sin_out], "SIN")
        self.convert_generic_tflite_op([pf.inputs[0]], [cos_out], "COS")
        self.convert_generic_tflite_op([sin_out, cos_out], pf.outputs, "DIV")
        self.propagate_variable_semantic(pf)

    def Reshape(self, pf):
        input = pf.inputs[0]
        shape = pf.args['shape']
        data_format = self.network.variables[pf.inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            input = self.convert_tensor_data_format_to_channel_first(input)
        self.generate_reshape_op(input, pf.outputs[0], shape)
        self.network.variables[pf.outputs[0]
                               ].data_format = self.global_data_format

    def BatchMatmul(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        a_shape = self.variables[inputs[0]]
        b_shape = self.variables[inputs[1]]
        out_shape = self.variables[outputs[0]]
        batch_dims_a = a_shape[:-2]
        batch_dims_b = b_shape[:-2]
        data_format_a = self.network.variables[inputs[0]].data_format
        data_format_b = self.network.variables[inputs[1]].data_format
        assert (data_format_a == data_format_b)
        self.network.variables[outputs[0]].data_format = data_format_a
        if batch_dims_a != batch_dims_b:
            # Broadcast
            const_one_a = fork_name(inputs[0]) + "_const_one_a"
            const_one_b = fork_name(inputs[1]) + "_const_one_b"
            batch_dims = [max(da, db)
                          for da, db in zip(batch_dims_a, batch_dims_b)]
            new_a_shape = batch_dims + a_shape[-2:]
            new_b_shape = batch_dims + b_shape[-2:]
            data_a = np.ones(new_a_shape, dtype=np.float32).flatten()
            data_b = np.ones(new_b_shape, dtype=np.float32).flatten()
            self.create_parameter(const_one_a, new_a_shape, data_a)
            self.create_parameter(const_one_b, new_b_shape, data_b)
            mul_out_a = fork_name(inputs[0]) + "_mul_a"
            mul_out_b = fork_name(inputs[1]) + "_mul_b"
            self.variables[mul_out_a] = new_a_shape
            self.variables[mul_out_b] = new_b_shape
            self.convert_generic_tflite_op(
                [inputs[0], const_one_a], [mul_out_a], "MUL")
            self.convert_generic_tflite_op(
                [inputs[1], const_one_b], [mul_out_b], "MUL")
            inputs = [mul_out_a, mul_out_b]
            a_shape = new_a_shape
            b_shape = new_b_shape

        # Reshape
        a_shape = [int(np.prod(a_shape[:-2]))] + a_shape[-2:]
        b_shape = [int(np.prod(b_shape[:-2]))] + b_shape[-2:]
        reshape_a = fork_name(inputs[0]) + "_reshape_a"
        reshape_b = fork_name(inputs[1]) + "_reshape_b"
        self.variables[reshape_a] = a_shape
        self.variables[reshape_b] = b_shape
        self.generate_reshape_op(inputs[0], reshape_a, a_shape)
        self.generate_reshape_op(inputs[1], reshape_b, b_shape)

        # BatchMatmul
        transpose_a = pf.args.get('transpose_a', False)
        transpose_b = pf.args.get('transpose_b', True)
        matmul_out = fork_name(outputs[0]) + "_matmul"
        self.variables[matmul_out] = [a_shape[0]] + out_shape[-2:]
        self.convert_generic_tflite_op([reshape_a, reshape_b], [
            matmul_out], "BATCH_MATMUL")
        self.operators_list[-1]['builtin_options_type'] = 'BatchMatMulOptions'
        self.operators_list[-1]['builtin_options'] = {
            'adj_x': transpose_a,
            'adj_y': transpose_b
        }

        # Reshape
        self.generate_reshape_op(matmul_out, outputs[0], out_shape)

    def Transpose(self, pf):
        input_shape = self.variables[pf.inputs[0]]
        axes = pf.args['axes']
        data_format = self.network.variables[pf.inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            new_axes = list(range(len(input_shape)))
            new_axes.insert(self.cur_base_axis, new_axes.pop(-1))
            new_axes = [new_axes[axes[i]] for i in range(len(input_shape))]
            axes = new_axes.copy()
            output_shape = [input_shape[axis] for axis in axes]
            self.variables[pf.outputs[0]] = output_shape

        axes_name = fork_name(pf.inputs[0]) + "_axes"
        self.create_parameter(axes_name, [len(axes)], axes)

        tensor_type = {axes_name: 'INT32'}
        self.convert_generic_tflite_op(
            [pf.inputs[0], axes_name], pf.outputs, "TRANSPOSE", tensor_type=tensor_type)
        self.network.variables[pf.outputs[0]
                               ].data_format = self.global_data_format

    def Concatenate(self, pf):
        is_converted, inputs, axis = self.check_resolve_axis_attr(pf)
        # Modify output shape
        if is_converted:
            output_shape = self.variables[inputs[0]].copy()
            output_shape[axis] = sum(
                [self.variables[inp][axis] for inp in inputs])
            self.variables[pf.outputs[0]] = output_shape
        self.convert_generic_tflite_op(inputs, pf.outputs, 'CONCATENATION')
        self.operators_list[-1]['builtin_options_type'] = 'ConcatenationOptions'
        self.operators_list[-1]['builtin_options'] = {'axis': axis}

    def Split(self, pf):
        is_converted, inputs, axis = self.check_resolve_axis_attr(pf)
        input_shape = self.variables[inputs[0]]
        # Modify output shape
        if is_converted:
            output_shape = input_shape.copy()
            del output_shape[axis]
            for outp in pf.outputs:
                self.variables[outp] = output_shape
        split_outs = []
        split_out_shape = input_shape.copy()
        split_out_shape[axis] = 1
        for _ in range(input_shape[axis]):
            out = fork_name(inputs[0]) + "_split"
            self.variables[out] = split_out_shape
            split_outs.append(out)
        self.generate_split_op(
            inputs[0], split_outs, axis, input_shape[axis])

        for i in range(input_shape[axis]):
            self.generate_reshape_op(
                split_outs[i], pf.outputs[i], self.variables[pf.outputs[i]])

    def Stack(self, pf):
        is_converted, inputs, axis = self.check_resolve_axis_attr(pf)
        # Modify output shape
        if is_converted:
            output_shape = self.variables[inputs[0]].copy()
            output_shape.append(len(inputs))
            self.variables[pf.outputs[0]] = output_shape
        self.convert_generic_tflite_op(inputs, pf.outputs, "PACK")
        self.operators_list[-1]['builtin_options_type'] = 'PackOptions'
        self.operators_list[-1]['builtin_options'] = {
            'values_count': len(inputs),
            'axis': axis,
        }

    def Slice(self, pf):
        input_shape = self.variables[pf.inputs[0]]
        output_shape = self.variables[pf.outputs[0]]
        starts = pf.args.get('start', [0] * len(input_shape))
        stops = pf.args.get('stop', input_shape)
        step = pf.args.get('step', [1] * len(input_shape))
        if step != [1] * len(step):
            raise ValueError('Currently, step != 1 not supported!')

        begin = [0] * (len(input_shape) - len(starts)) + starts
        size = input_shape[:(len(input_shape) - len(starts))] + \
            [stops[i] - starts[i] for i in range(len(starts))]
        data_format = self.network.variables[pf.inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            begin.append(begin.pop(self.cur_base_axis))
            size.append(size.pop(self.cur_base_axis))
            output_shape.append(output_shape.pop(self.cur_base_axis))
            self.variables[pf.outputs[0]] = output_shape
            self.network.variables[pf.outputs[0]
                                   ].data_format = DataFormat.channel_last
        else:
            self.network.variables[pf.outputs[0]].data_format = data_format
        begin_name = fork_name(pf.inputs[0]) + "_begin"
        size_name = fork_name(pf.inputs[0]) + "_size"
        self.create_parameter(begin_name, [len(begin)], begin)
        self.create_parameter(size_name, [len(size)], size)
        tensor_type = {begin_name: 'INT32', size_name: 'INT32'}
        self.convert_generic_tflite_op(
            [pf.inputs[0], begin_name, size_name], pf.outputs, "SLICE", tensor_type=tensor_type)

    def Pad(self, pf):
        mode = pf.args.get('mode', 'constant')
        pad_width = pf.args['pad_width']
        constant_value = pf.args.get('constant_value', 0)
        input_shape = self.variables[pf.inputs[0]]
        output_shape = self.variables[pf.outputs[0]]
        diff = len(input_shape) - (len(pad_width) // 2)
        pad_width = [0] * 2 * diff + pad_width
        padding = list(zip(pad_width[::2], pad_width[1::2]))
        data_format = self.network.variables[pf.inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            padding.append(padding.pop(self.cur_base_axis))
            output_shape.append(output_shape.pop(self.cur_base_axis))
            self.variables[pf.outputs[0]] = output_shape
            self.network.variables[pf.outputs[0]
                                   ].data_format = DataFormat.channel_last
        else:
            self.network.variables[pf.outputs[0]].data_format = data_format
        padding = np.array(padding, dtype=np.int32)
        padding_name = fork_name(pf.inputs[0]) + "_padding"
        self.create_parameter(
            padding_name, [len(padding), 2], padding.flatten().tolist())
        tensor_type = {padding_name: 'INT32'}
        if mode == 'constant':
            value_name = fork_name(pf.inputs[0]) + "_value"
            self.create_parameter(value_name, [1], [constant_value])
            self.convert_generic_tflite_op(
                [pf.inputs[0], padding_name, value_name], pf.outputs, 'PADV2', tensor_type=tensor_type)
        elif mode == 'reflect':
            self.convert_generic_tflite_op(
                [pf.inputs[0], padding_name], pf.outputs, 'MIRROR_PAD', tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'MirrorPadOptions'
            self.operators_list[-1]['builtin_options'] = {'mode': 0}

    def Tile(self, pf):
        input_shape = self.variables[pf.inputs[0]]
        reps = pf.args['reps']
        input = pf.inputs[0]
        if len(reps) > len(input_shape):
            new_input_shape = [1] * \
                (len(reps) - len(input_shape)) + input_shape
            input_reshape = fork_name(input) + "_reshape"
            self.variables[input_reshape] = new_input_shape
            self.generate_reshape_op(input, input_reshape, new_input_shape)
            input = input_reshape
        else:
            reps = [1] * (len(input_shape) - len(reps)) + reps
        reps_name = fork_name(pf.inputs[0]) + "_reps"
        self.create_parameter(reps_name, [len(reps)], reps)
        tensor_type = {reps_name: 'INT32'}
        self.convert_generic_tflite_op(
            [input, reps_name], pf.outputs, 'TILE', tensor_type=tensor_type)
        self.propagate_variable_semantic(pf)

    def Flip(self, pf):
        inputs = pf.inputs[:]
        input_shape = self.variables[pf.inputs[0]]
        output_shape = self.variables[pf.outputs[0]]
        axes = pf.args.get('axes', [len(input_shape) - 1])

        # create intermediate output node
        outs = []
        for _ in range(len(axes) - 1):
            o = fork_name(inputs[0]) + "_transpose"
            self.variables[o] = output_shape
            outs.append(o)
        outs.extend(pf.outputs)

        for i, axis in enumerate(axes):
            s = np.arange(len(input_shape))
            # Step 1: transpose
            perm = np.roll(s, -axis)
            transpose_out = fork_name(inputs[0]) + "_transpose"
            self.variables[transpose_out] = np.roll(
                input_shape, -axis).tolist()
            axes_name = fork_name(inputs[0]) + "_axes"
            self.create_parameter(axes_name, [len(perm)], perm)

            tensor_type = {axes_name: 'INT32'}
            self.convert_generic_tflite_op(
                [inputs[0], axes_name], [transpose_out], "TRANSPOSE", tensor_type=tensor_type)

            # Step 2: Gather
            positions_name = fork_name(inputs[0]) + "_positions"
            gather_out = fork_name(inputs[0]) + "_gather"
            raw_data = np.arange(input_shape[axis])[
                ::-1].astype(np.int32).flatten().tolist()
            positions_shape = [input_shape[axis]]
            self.create_parameter(positions_name, positions_shape, raw_data)
            self.variables[gather_out] = self.variables[transpose_out]
            tensor_type = {positions_name: 'INT32'}
            self.convert_generic_tflite_op([transpose_out, positions_name], [
                                           gather_out], 'GATHER', tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'GatherOptions'
            self.operators_list[-1]['builtin_options'] = {'axis': 0}

            # Step 3: transpose
            perm = np.roll(s, axis)
            axes_name = fork_name(inputs[0]) + "_axes"
            self.create_parameter(axes_name, [len(perm)], perm)
            tensor_type = {axes_name: 'INT32'}
            self.convert_generic_tflite_op(
                [gather_out, axes_name], [outs[i]], "TRANSPOSE", tensor_type=tensor_type)
            inputs[0] = outs[i]
        self.propagate_variable_semantic(pf)

    def Broadcast(self, pf):
        shape = [self.batch_size if d < 0 else d for d in pf.args['shape']]
        if len(shape) > 4:
            raise ValueError(
                'Currently, Dimensions > 4 will not be supported.')
        data_format = self.network.variables[pf.inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            shape.append(shape.pop(self.cur_base_axis))
            self.variables[pf.outputs[0]] = shape
            self.network.variables[pf.outputs[0]
                                   ].data_format = DataFormat.channel_last
        else:
            self.network.variables[pf.outputs[0]].data_format = data_format
        const_one = fork_name(pf.inputs[0]) + "_const_one"
        data = np.ones(shape, dtype=np.float32).flatten()
        self.create_parameter(const_one, shape, data)
        self.convert_generic_tflite_op(
            [pf.inputs[0], const_one], pf.outputs, 'MUL')

    def ReductionOperator(self, tflite_operator, pf):
        inputs = pf.inputs[:]
        input_shape = self.variables[inputs[0]]
        axes = pf.args.get('axes', list(range(len(input_shape))))
        keep_dims = pf.args.get('keep_dims', False)

        data_format = self.network.variables[pf.inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            inputs[0] = self.convert_tensor_data_format_to_channel_first(
                inputs[0])

        axes_name = fork_name(inputs[0]) + "_axes"
        self.create_parameter(axes_name, [len(axes)], axes)

        tensor_type = {axes_name: 'INT32'}
        self.convert_generic_tflite_op(
            [inputs[0], axes_name], pf.outputs, tflite_operator, tensor_type=tensor_type)
        self.operators_list[-1]['builtin_options_type'] = 'ReducerOptions'
        self.operators_list[-1]['builtin_options'] = {'keep_dims': keep_dims}
        self.network.variables[pf.outputs[0]
                               ].data_format = self.global_data_format

    def ScalarOperator(self, tflite_operator, pf):
        reverse = False
        if pf.type in ['RDivScalar', 'RPowScalar', 'RSubScalar']:
            reverse = True
        value = pf.args.get('val', 1.0)
        scalar_name = fork_name(pf.inputs[0]) + "_scalar"
        self.create_parameter(scalar_name, [1], [value])
        if reverse:
            inputs = [scalar_name, pf.inputs[0]]
        else:
            inputs = [pf.inputs[0], scalar_name]
        self.convert_generic_tflite_op(inputs, pf.outputs, tflite_operator)
        self.propagate_variable_semantic(pf)

    def BatchNormalization(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        eps = pf.args.get('eps', 1e-05)
        axes = pf.args.get('axes', (1,))[0]
        batch_stat = pf.args.get('batch_stat', True)

        data_format = self.network.variables[inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            for pv in inputs[1:5]:
                self.convert_parameter_data_format(pv)
            if axes == self.cur_base_axis:
                axes = len(self.variables[pf.inputs[0]]) - 1
            else:
                axes = axes - 1 if axes > self.cur_base_axis else axes
            self.variables[outputs[0]].append(
                self.variables[outputs[0]].pop(self.cur_base_axis))
            self.network.variables[outputs[0]
                                   ].data_format = DataFormat.channel_last
        else:
            self.network.variables[outputs[0]].data_format = data_format

        self.generate_bn_op(inputs, outputs, eps, axes, batch_stat)

    def FusedBatchNormalization(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        batch_stat = pf.args.get('batch_stat', True)
        nonlinearity = pf.args.get('nonlinearity', 'relu')
        eps = pf.args.get('eps', 1e-05)
        axes = pf.args.get('axes', (1,))[0]
        input_shape = self.variables[inputs[0]]

        data_format = self.network.variables[inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            for pv in inputs[1:5]:
                self.convert_parameter_data_format(pv)
            if axes == self.cur_base_axis:
                axes = len(self.variables[pf.inputs[0]]) - 1
            else:
                axes = axes - 1 if axes > self.cur_base_axis else axes
            self.variables[outputs[0]].append(
                self.variables[outputs[0]].pop(self.cur_base_axis))
            self.network.variables[outputs[0]
                                   ].data_format = DataFormat.channel_last
        else:
            self.network.variables[outputs[0]].data_format = data_format

        bn_out = fork_name(inputs[0]) + "_bn"
        self.variables[bn_out] = input_shape
        self.generate_bn_op(inputs, outputs, eps, axes, batch_stat)

        if len(inputs) > 5:
            # Add
            data_format_z = self.network.variables[inputs[5]].data_format
            assert (data_format_z == data_format)
            add_out = fork_name(inputs[0]) + "_add"
            self.variables[add_out] = input_shape
            self.convert_generic_tflite_op(
                [bn_out, inputs[5]], [add_out], 'ADD')

        if nonlinearity == "relu":
            # Here, we assume the last operator is 'Add'. Because it is lucky that
            # previous operator only can be Add, although this assumption is vulnerable.
            self.operators_list[-1]['builtin_options_type'] = 'AddOptions'
            self.operators_list[-1]['builtin_options'] = {
                'fused_activation_function': 'RELU'}
        else:
            raise ValueError(
                "Currently, nonlinearity != relu is not supported!")

    def Affine(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        weight_shape = self.variables[inputs[1]]
        base_axis = pf.args.get('base_axis', 1)
        if inputs[1] not in self.resolved_parameters:
            weight = self.parameters[inputs[1]]
            var_d = np.array(
                weight.data, dtype=np.float32).reshape(weight_shape)
            new_weight_shape = np.array(
                weight_shape[:1] + [np.prod(weight_shape[1:])]).flatten().tolist()
            new_var_d = np.reshape(var_d, new_weight_shape)
            new_var_d = np.transpose(new_var_d, [1, 0]).flatten()
            self.create_parameter(inputs[1], list(
                reversed(new_weight_shape)), new_var_d)
            self.variables[inputs[1]] = list(reversed(new_weight_shape))
            self.resolved_parameters.append(inputs[1])

        data_format = self.network.variables[inputs[0]].data_format
        if self.global_data_format == DataFormat.channel_first and \
                data_format == DataFormat.channel_last:
            inputs[0] = self.convert_tensor_data_format_to_channel_first(
                inputs[0])

        input_shape = self.variables[inputs[0]]
        output_shape = self.variables[outputs[0]]
        new_input_shape = np.array([np.prod(input_shape[:base_axis]), np.prod(
            input_shape[base_axis:])]).flatten().tolist()
        new_output_shape = np.array([np.prod(output_shape[:base_axis]), np.prod(
            output_shape[base_axis:])]).flatten().tolist()
        before_reshape = fork_name(inputs[0]) + "_reshape"
        after_reshape = fork_name(outputs[0]) + "_reshape"
        self.variables[before_reshape] = new_input_shape
        self.variables[after_reshape] = new_output_shape
        self.generate_reshape_op(
            inputs[0], before_reshape, new_input_shape)
        self.generate_reshape_op(
            after_reshape, outputs[0], self.variables[outputs[0]])
        inputs[0] = before_reshape
        outputs[0] = after_reshape

        if len(inputs) > 2:
            bias_shape = self.variables[inputs[2]]
            new_bias_shape = np.array([np.prod(bias_shape)]).flatten().tolist()
            self.parameters[inputs[2]].shape.Clear()
            self.parameters[inputs[2]].shape.dim.extend(new_bias_shape)
            self.variables[inputs[2]] = new_bias_shape
        self.convert_generic_tflite_op(inputs, outputs, 'FULLY_CONNECTED')
        self.operators_list[-1]['builtin_options_type'] = 'FullyConnectedOptions'
        self.operators_list[-1]['builtin_options'] = {}
        self.network.variables[pf.outputs[0]
                               ].data_format = self.global_data_format

    def Interpolate(self, pf):
        inputs = pf.inputs[:]
        outputs = pf.outputs[:]
        input_shape = self.variables[inputs[0]]
        output_shape = self.variables[outputs[0]]
        mode = pf.args['mode']
        output_size = pf.args['output_size']
        align_corners = pf.args.get('align_corners', True)
        half_pixel = pf.args.get('half_pixel', False)
        half_pixel_for_nn = pf.args.get('half_pixel_for_nn', False)

        if len(output_size) != 2:
            raise ValueError(
                "Currently, only output_size == 2 is supported!")
        if len(input_shape) != 4:
            raise ValueError(
                "Currently, only 4D is supported!")
        if half_pixel:
            raise ValueError(
                "Currently, Does not support half_pixel as True!")

        data_format = self.network.variables[pf.inputs[0]].data_format
        if data_format == DataFormat.channel_first:
            inputs[0] = self.convert_tensor_data_format_to_channel_last(
                inputs[0])
            self.variables[outputs[0]] = output_shape.append(
                output_shape.pop(1))

        output_size_name = fork_name(pf.inputs[0]) + "_size"
        self.create_parameter(output_size_name, [2], output_size)
        inputs.append(output_size_name)
        tensor_type = {output_size_name: 'INT32'}

        if mode == 'nearest':
            if half_pixel_for_nn:
                tflite_half_pixel = True
                tflite_align_corners = False
            else:
                if align_corners:
                    raise ValueError(
                        "Currently, nearest mode does not support align_corners as True!")
                tflite_align_corners = False
                tflite_half_pixel = False
            self.convert_generic_tflite_op(
                inputs, outputs, 'RESIZE_NEAREST_NEIGHBOR', tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'ResizeNearestNeighborOptions'
            self.operators_list[-1]['builtin_options'] = {
                'align_corners': tflite_align_corners,
                'half_pixel_centers': tflite_half_pixel,
            }
        else:
            self.convert_generic_tflite_op(
                inputs, outputs, 'RESIZE_BILINEAR', tensor_type=tensor_type)
            self.operators_list[-1]['builtin_options_type'] = 'ResizeBilinearOptions'
            self.operators_list[-1]['builtin_options'] = {
                'align_corners': align_corners,
                'half_pixel_centers': False,
            }
        self.network.variables[pf.outputs[0]
                               ].data_format = DataFormat.channel_last

    def create_model_buffer(self, name, dtype):
        if name in self.parameters:
            buffer_data = {}
            # Get data of parameter
            param = self.parameters[name]
            var_d = np.array(param.data, dtype=dtype)
            shape = list(param.shape.dim)
            var_d = np.reshape(var_d, shape)
            bytes = var_d.tobytes()
            buffer_data['data'] = [bytes[i] for i in range(len(bytes))]
        else:  # variable.type == 'Buffer'
            buffer_data = {}
        self.models['buffers'].append(buffer_data)
        self.buffer_list.append(name)

    def create_tensor_info(self, shape, buffer_index, name, quantization, dtype, shape_signature=None):
        tensor_info = dict()
        tensor_info['shape'] = shape
        tensor_info['buffer'] = buffer_index
        tensor_info['name'] = name
        tensor_info['quantization'] = quantization
        tensor_info['type'] = dtype
        if shape_signature is not None:
            tensor_info['shape_signature'] = shape_signature
        return tensor_info

    def create_tensors(self, var_list, tensor_type):
        TENSOR_TYPE_TO_DTYPE = {
            'FLOAT32': np.float32,
            'INT32': np.int32,
            'BOOL': bool,
        }
        for name in var_list:
            dtype = tensor_type[name] if name in tensor_type else 'FLOAT32'
            if name in self.tensor_list:
                # Correct the tensor shape
                index = self.tensor_list.index(name)
                tensor_info = self.models['subgraphs'][0]['tensors'][index]
                tensor_info['shape'] = self.variables[name]
                tensor_info['type'] = dtype
            else:
                self.create_model_buffer(name, TENSOR_TYPE_TO_DTYPE[dtype])
                var_shape = self.variables[name]
                var_buffer_index = self.buffer_list.index(name)
                var_quantization = {}
                tensor_info = self.create_tensor_info(
                    var_shape, var_buffer_index, name, var_quantization, dtype)
                self.models['subgraphs'][0]['tensors'].append(tensor_info)
                self.tensor_list.append(name)

            if name in self.inputs:
                inp_index = self.inputs.index(name)
                self.models['subgraphs'][0]['inputs'][inp_index] = self.tensor_list.index(
                    name)

            if name in self.outputs:
                oup_index = self.outputs.index(name)
                self.models['subgraphs'][0]['outputs'][oup_index] = self.tensor_list.index(
                    name)

    def create_operator(self, inputs, outputs, tflite_operator_codes):
        operator = dict()
        if tflite_operator_codes not in self.operator_codes_list:
            self.models['operator_codes'].append({
                "builtin_code": tflite_operator_codes,
                "version": 1
            })
            self.operator_codes_list.append(tflite_operator_codes)
        operator['opcode_index'] = self.operator_codes_list.index(
            tflite_operator_codes)
        operator['inputs'] = [self.tensor_list.index(inp) for inp in inputs]
        operator['outputs'] = [self.tensor_list.index(oup) for oup in outputs]
        self.operators_list.append(operator)

    def convert_generic_tflite_op(self, inputs, outputs, tflite_operator_codes, tensor_type={}):
        # create buffer
        tensor_list = inputs[:] + outputs[:]
        self.create_tensors(tensor_list, tensor_type)

        self.create_operator(inputs, outputs, tflite_operator_codes)

    def convert_to_tflite_op(self, func):
        '''
        Convert NNabla function to tflite op.
        '''
        tflite_operator_codes = self.map_nn_func_to_tflite_op.get(
            func.type, None)
        if not tflite_operator_codes:
            raise ValueError(
                "function {} is currently not supported for TFLite conversion".format(func.type))

        if callable(tflite_operator_codes):
            return tflite_operator_codes(func)

        return self.convert_generic_tflite_op(func.input, func.output, tflite_operator_codes)

    def set_variables(self):
        '''
        load all variable and parameter.
        '''
        bs = self.batch_size
        if bs < 0:
            bs = self.network.batch_size
        self.batch_size = bs

        for name, pv in self.network.variables.items():
            self.variables[name] = replace_negative_size_with_batch_size(
                pv.shape, bs)

        for p in self.nnp_proto.parameter:
            self.variables[p.variable_name] = list(p.shape.dim)
            self.parameters[p.variable_name] = p

        for gv in self.executor.generator_variable:
            shape = self.variables[gv.variable_name]
            data = generate_value(gv.type, shape, gv.multiplier)
            self.create_parameter(gv.variable_name, shape, data)

        for inp in self.network.inputs:
            if inp not in self.parameters:
                self.inputs.append(inp)

        self.outputs.extend(self.network.outputs)

    def set_network(self):
        '''
        Get network and exector from nnp_proto.
        '''
        if len(self.nnp_proto.executor) != 1:
            raise ValueError(
                "NNP with only a single executor is currently supported")
        exe = self.nnp_proto.executor[0]
        self.g = nn.graph_def.ProtoGraph.from_proto(
            self.nnp_proto, batch_size=self.batch_size)
        self.network = self.g.networks.get(exe.network_name, None)
        self.network.execute_on_proto(IdentityRemover({}))
        self.executor = exe

        if self.network is None:
            raise ValueError(
                "Executor network [{}] is not found in this NNP.".format(exe.network_name))

    def init_models(self):
        '''
        Initialize tflite model dict.
        '''
        models_info = dict()
        subgraph_info = dict()
        subgraph_info['tensors'] = []
        subgraph_info['inputs'] = []
        subgraph_info['outputs'] = []
        subgraph_info['operators'] = []
        subgraph_info['name'] = 'main'
        models_info["version"] = 3
        models_info['operator_codes'] = []
        models_info['subgraphs'] = [subgraph_info]
        models_info['description'] = "MLIR Converted."
        models_info['buffers'] = [
            {"data": [49, 46, 49, 49, 46, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]
        models_info['metadata'] = [{
            "name": "min_runtime_version",
            "buffer": 0}]
        self.models = models_info

    def check_and_preprocess_input(self):
        # If format is NCHW, the inputs will be preprocessed.
        # The input variable is referenced by the implicit node,
        # the shape of the input variable will be directly modified, and the base_axis information will be saved.
        implicit_list = ['Convolution', 'DepthwiseConvolution', 'Deconvolution', 'DepthwiseDeconvolution',
                         'MaxPooling', 'AveragePooling', 'SumPooling', 'Unpooling']
        if self.global_data_format == DataFormat.channel_first:
            for inp in self.inputs:
                for func in self.network.variables[inp].required:
                    f_type = self.network.functions[func].type
                    if f_type in implicit_list:
                        pf = self.network.functions[func]
                        if 'base_axis' in pf.args:
                            base_axis = pf.args['base_axis']
                        else:
                            base_axis = len(
                                self.variables[inp]) - len(pf.args['kernel']) - 1
                        self.cur_base_axis = base_axis
                        self.variables[inp].append(
                            self.variables[inp].pop(base_axis))
                        self.network.variables[inp].data_format = DataFormat.channel_last
                        self.tflite_model_info['inputs'][inp] = {
                            'transpose': True, 'base_axis': base_axis}
                    else:
                        self.tflite_model_info['inputs'][inp] = {
                            'transpose': False}
        else:
            for inp in self.inputs:
                self.network.variables[inp].data_format = DataFormat.channel_last
                self.tflite_model_info['inputs'][inp] = {'transpose': False}

    def export_graph(self):
        self.set_network()
        self.set_variables()
        for inp in self.network.inputs:
            self.network.variables[inp].data_format = self.global_data_format
        self.models['subgraphs'][0]['inputs'].extend([-1] * len(self.inputs))
        self.models['subgraphs'][0]['outputs'].extend([-1] * len(self.outputs))
        # create inputs and outputs tensors
        self.create_tensors(self.inputs, {})
        self.create_tensors(self.outputs, {})
        self.check_and_preprocess_input()
        precursor_function = None
        for pfid, pf in enumerate(self.network.forward_sequence()):
            self.child_of_relu = False
            if pfid > 0 and pf.type == 'MaxPooling':
                if precursor_function is not None and precursor_function.type == 'ReLU':
                    self.child_of_relu = True
            self.convert_to_tflite_op(pf)
            self.operators_sorting(pf)
            precursor_function = pf
        if self.global_data_format == DataFormat.channel_first:
            for outp in self.outputs:
                if self.network.variables[outp].data_format == DataFormat.channel_last:
                    self.tflite_model_info['outputs'][outp] = {
                        'transpose': True, 'base_axis': self.cur_base_axis}
                else:
                    self.tflite_model_info['outputs'][outp] = {
                        'transpose': False}
        else:
            for outp in self.outputs:
                self.tflite_model_info['outputs'][outp] = {'transpose': False}

        # Remove useless inputs
        self.models['subgraphs'][0]['inputs'] = [
            index for index in self.models['subgraphs'][0]['inputs'] if index != -1]

    def get_operator_name(self, operator):
        opcode_index = operator['opcode_index']
        op_name = self.models['operator_codes'][opcode_index]['builtin_code']
        return op_name

    def get_number_of_children_op(self, output):
        children_number = 0
        for child_operator in self.models['subgraphs'][0]['operators']:
            inputs_of_child_operator = child_operator['inputs']
            if output in inputs_of_child_operator:
                children_number += 1
        return children_number

    def fuse_activation_function(self):
        ops_can_fused_with_act = ['CONV_2D', 'ADD']
        acts_can_be_fused = ['RELU']
        for operator in self.models['subgraphs'][0]['operators']:
            op_name = self.get_operator_name(operator)
            if op_name in ops_can_fused_with_act:
                outpus = operator['outputs']
                if len(outpus) > 1:
                    continue
                children_number = self.get_number_of_children_op(outpus[0])
                if children_number == 0 or children_number > 1:  # Leaf node or has more than one children
                    continue
                for operator_index, child_operator in enumerate(self.models['subgraphs'][0]['operators']):
                    inputs_of_child_operator = child_operator['inputs']
                    op_name_of_child_operator = self.get_operator_name(
                        child_operator)
                    if outpus[0] in inputs_of_child_operator and op_name_of_child_operator in acts_can_be_fused:
                        # fuse act function
                        outputs_of_child_operator = child_operator['outputs']
                        # search the child of child node
                        for grandson_operator in self.models['subgraphs'][0]['operators']:
                            inputs_of_grandson_op = grandson_operator['inputs']
                            # set grandfather's output as grandson's input
                            if outputs_of_child_operator[0] in inputs_of_grandson_op:
                                grandson_operator['inputs'][inputs_of_grandson_op.index(
                                    outputs_of_child_operator[0])] = outpus[0]

                        # set grandfather's fused_activate_function
                        if operator.get('builtin_options') is None:
                            if op_name == 'CONV_2D':
                                operator['builtin_options_type'] = 'Conv2DOptions'
                            if op_name == 'ADD':
                                operator['builtin_options_type'] = 'AddOptions'
                            operator['builtin_options'] = {}
                        operator['builtin_options']['fused_activation_function'] = op_name_of_child_operator

                        # pop activation op
                        self.models['subgraphs'][0]['operators'].pop(
                            operator_index)

                        # pop output tensor and buffer of act op
                        for act_output in outputs_of_child_operator:
                            tensor_of_act = self.models['subgraphs'][0]['tensors'].pop(
                                act_output)
                            buffer_index_of_act = tensor_of_act['buffer']
                            self.models['buffers'].pop(buffer_index_of_act)
                            for tensor in self.models['subgraphs'][0]['tensors']:
                                buffer_index = tensor['buffer']
                                if buffer_index > buffer_index_of_act:
                                    tensor['buffer'] = buffer_index - 1
                            for _operator in self.models['subgraphs'][0]['operators']:
                                _inputs = _operator['inputs']
                                updated_inputs = []
                                for tensor_index in _inputs:
                                    if tensor_index > act_output:
                                        tensor_index -= 1
                                    updated_inputs.append(tensor_index)
                                _operator['inputs'] = updated_inputs
                                _outputs = _operator['outputs']
                                updated_outputs = []
                                for tensor_index in _outputs:
                                    if tensor_index > act_output:
                                        tensor_index -= 1
                                    updated_outputs.append(tensor_index)
                                _operator['outputs'] = updated_outputs
                                _outputs = self.models['subgraphs'][0]['outputs']
                                updated_outputs = []
                                for tensor_index in _outputs:
                                    if tensor_index > act_output:
                                        tensor_index -= 1
                                    updated_outputs.append(tensor_index)
                                self.models['subgraphs'][0]['outputs'] = updated_outputs

    def optimize_graph(self):
        self.fuse_activation_function()

    def mark_all_buffers_as_output(self):
        import copy
        self.backup_output = copy.copy(self.models['subgraphs'][0]['outputs'])
        for operator in self.models['subgraphs'][0]['operators']:
            outputs = operator['outputs']
            for output in outputs:
                if output not in self.models['subgraphs'][0]['outputs']:
                    self.models['subgraphs'][0]['outputs'].append(output)
        inputs = self.models['subgraphs'][0]['inputs']
        for inp in inputs:
            if inp not in self.models['subgraphs'][0]['outputs']:
                self.models['subgraphs'][0]['outputs'].append(inp)

    def export_to_tflite(self, models, output):
        """
        :param output (str): file path of exported tflite
        :return:
        """
        json_file = output.replace("tflite", "json")
        with open(json_file, 'w') as f:
            json.dump(models, f)
        schema_path = os.path.join(os.path.dirname(__file__), "schema.fbs")
        output_path = os.path.dirname(output)
        try:
            subprocess.check_output(
                [flatc_path, '-b', '-o', output_path, schema_path, json_file], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.returncode, e.output)
            raise ValueError("Convert nnp to tflite failed.")

    def execute(self, output):
        self.init_models()

        self.export_graph()

        self.optimize_graph()

        if self.quantization:
            self.mark_all_buffers_as_output()

        self.export_to_tflite(self.models, output)

        if self.quantization:
            from .quantized_converter import QuantizationConverter
            dataset = np.load(self.dataset)
            quantization_converter = QuantizationConverter(
                self.models, output, dataset)
            quantized_model = quantization_converter.convert()
            quantized_model['subgraphs'][0]['outputs'] = self.backup_output
            self.export_to_tflite(quantized_model, output)

        json_file = output.replace("tflite", "json")
        with open(json_file, 'w') as f:
            json.dump(self.tflite_model_info, f)
