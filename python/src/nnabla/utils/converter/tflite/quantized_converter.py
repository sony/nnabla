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
import tensorflow as tf

# OPs that support INT8 data type
QUANTIZATION_OPS = [
    'ADD',
    'AVERAGE_POOL_2D',
    'CONCATENATION',
    'CONV_2D',
    'DEPTHWISE_CONV_2D',
    'FULLY_CONNECTED',
    'L2_NORMALIZATION',
    'LOGISTIC',
    'MAX_POOL_2D',
    'MUL',
    'RESHAPE',
    'RESIZE_BILINEAR',
    'SOFTMAX',
    'SPACE_TO_DEPTH',
    'TANH',
    'PAD',
    'GATHER',
    'BATCH_TO_SPACE_ND',
    'SPACE_TO_BATCH_ND',
    'TRANSPOSE',
    'MEAN',
    'SUB',
    'SUM',
    'SQUEEZE',
    'LOG_SOFTMAX',
    'MAXIMUM',
    'ARG_MAX',
    'MINIMUM',
    'LESS',
    'PADV2',
    'GREATER',
    'GREATER_EQUAL',
    'LESS_EQUAL',
    'SLICE',
    'EQUAL',
    'NOT_EQUAL',
    'SHAPE',
    'QUANTIZE',
    'RELU',
    'LEAKY_RELU'
]


class QuantizeGranularity(object):
    PER_TENSOR = 0
    PER_AXIS_DIM0 = 1
    PER_AXIS_DIM3 = 2


class Restriction(object):
    ZERO_POINT_TO_ZERO = 0
    CONV_2D_BIAS = 1
    SAME_WITH_INPUT = 2
    SOFTMAX = 3
    CONCAT = 4


# Constant
MAX_INT8 = 127
MIN_INT8 = -128
MAX_INT32 = 2**31-1
MIN_INT32 = -2**31


def construct_requirement(dtype, granularity, restriction=None, value_range=(MIN_INT8, MAX_INT8)):
    d = {}
    d['data_type'] = dtype
    d['granularity'] = granularity
    d['restriction'] = restriction
    d['range'] = value_range
    return d


# Reference: https://tensorflow.google.cn/lite/performance/quantization_spec
QUANTIZATIONREQUIREMENTS = {
    'ADD': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
            'Input_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, value_range=(MIN_INT8+1, MAX_INT8)),
            'Output_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
            'op_version': 2},
    'AVERAGE_POOL_2D': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                        'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                        'op_version': 2},
    'CONCATENATION': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                      'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.CONCAT),
                      'op_version': 2},
    'CONV_2D': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'Input_1': construct_requirement('int8', QuantizeGranularity.PER_AXIS_DIM0, Restriction.ZERO_POINT_TO_ZERO, (MIN_INT8+1, MAX_INT8)),
                'Input_2': construct_requirement('int32', QuantizeGranularity.PER_AXIS_DIM0, Restriction.CONV_2D_BIAS, (MIN_INT32, MAX_INT32)),
                'op_version': 3},
    'RELU': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
             'op_version': 2},
    'LEAKY_RELU': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                   'op_version': 2},
    'DEPTHWISE_CONV_2D': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                          'Input_1': construct_requirement('int8', QuantizeGranularity.PER_AXIS_DIM3, Restriction.ZERO_POINT_TO_ZERO, (MIN_INT8+1, MAX_INT8)),
                          'Input_2': construct_requirement('int32', QuantizeGranularity.PER_AXIS_DIM3, Restriction.CONV_2D_BIAS),
                          'op_version': 3},
    'FULLY_CONNECTED': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                        'Input_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.ZERO_POINT_TO_ZERO, (MIN_INT8+1, MAX_INT8)),
                        'Input_2': construct_requirement('int32', QuantizeGranularity.PER_TENSOR, Restriction.CONV_2D_BIAS, (MIN_INT32, MAX_INT32)),
                        'op_version': 4},
    'L2_NORMALIZATION': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                         'op_version': 4},
    'LOGISTIC': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                 'op_version': 2},
    'MAX_POOL_2D': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                    'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                    'op_version': 2},
    'MUL': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
            'Input_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, value_range=(MIN_INT8+1, MAX_INT8)),
            'op_version': 2},
    'RESHAPE': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT)},
    'RESIZE_BILINEAR': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                        'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                        'op_version': 2},
    'SOFTMAX': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SOFTMAX),
                'op_version': 2},
    'SPACE_TO_DEPTH': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                       'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                       'op_version': 2},
    'TANH': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
             'op_version': 2},
    'PAD': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
            'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
            'op_version': 2},
    'GATHER': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
               'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
               'op_version': 2},
    'BATCH_TO_SPACE_ND': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                          'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                          'op_version': 2},
    'SPACE_TO_BATCH_ND': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                          'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                          'op_version': 2},
    'TRANSPOSE': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                  'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                  'op_version': 2},
    'MEAN': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
             'op_version': 2},
    'SUB': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
            'Input_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
            'op_version': 2},
    'SUM': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
            'op_version': 2},
    'SQUEEZE': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT)},
    'LOG_SOFTMAX': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                    'op_version': 2},
    'MAXIMUM': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                'op_version': 2},
    'ARG_MAX': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'op_version': 2},
    'MINIMUM': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
                'op_version': 2},
    'LESS': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
             'Input_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
             'op_version': 2},
    'PADV2': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
              'Input_2': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
              'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
              'op_version': 2},
    'GREATER': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                'op_version': 2},
    'GREATER_EQUAL': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                      'Input_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                      'op_version': 2},
    'LESS_EQUAL': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                   'Input_1': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                   'op_version': 2},
    'SLICE': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
              'Output_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR, Restriction.SAME_WITH_INPUT),
              'op_version': 2},
    'EQUAL': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
              'op_version': 2},
    'NOT_EQUAL': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR),
                  'op_version': 2},
    'SHAPE': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR)},
    'QUANTIZE': {'Input_0': construct_requirement('int8', QuantizeGranularity.PER_TENSOR)},
}

DTYPES = {
    'INT8': np.int8,
    'INT32': np.int32,
    'BOOL': bool
}


def check_opset(models):
    """
    :param models (dict): dictionary hat contains the information of tflite model
    :return: Te or False. The returned value means whether this tflite model can be quantized or not
    """
    for op_code in models['operator_codes']:
        if not op_code.get('builtin_code'):
            op_code['builtin_code'] = 'ADD'
        if op_code['builtin_code'] not in QUANTIZATION_OPS:
            raise ValueError("Can't quantize this model because it includes unsupported op {}".format(
                op_code['builtin_code']))
    return True


class QuantizationConverter(object):
    def __init__(self, models, tflite, dataset):
        self.models = models
        self.calibrator = Calibrator(tflite, dataset)

    def get_tensor_data_by_index(self, tensor_idx):
        for i, tensor in enumerate(self.models['subgraphs'][0]['tensors']):
            if i == tensor_idx:
                if not tensor.get('type'):
                    tensor['type'] = 'FLOAT32'
                dtype = DTYPES.get(tensor['type'], np.float32)
                buffer_idx = tensor['buffer']
                data = self.models['buffers'][buffer_idx]['data']
                data = bytearray(data)  # list to bytearray
                data = np.frombuffer(data, dtype=dtype)  # bytearray to ndarray
                data = np.reshape(data, tensor['shape'])
                return data

    def get_tensor_type_by_index(self, tensor_idx):
        for i, tensor in enumerate(self.models['subgraphs'][0]['tensors']):
            if i == tensor_idx:
                buffer_idx = tensor['buffer']
                data = self.models['buffers'][buffer_idx]
                if bool(data):
                    return 'Parameter'
                else:
                    return 'Buffer'

    def get_tensor_by_index(self, tensor_idx):
        for i, tensor in enumerate(self.models['subgraphs'][0]['tensors']):
            if i == tensor_idx:
                return tensor

    def get_tensor_and_data_by_index(self, tensor_idx):
        return self.get_tensor_data_by_index(tensor_idx), self.get_tensor_by_index(tensor_idx)

    def set_tensor_data_by_index(self, tensor_idx, data):
        for i, tensor in enumerate(self.models['subgraphs'][0]['tensors']):
            if i == tensor_idx:
                buffer_idx = tensor['buffer']
                self.models['buffers'][buffer_idx]['data'] = data

    def quantize_buffer(self):
        # Quantize buffer by calibrator
        quantization_param = self.calibrator.run()
        for buffer_name in quantization_param:
            for tensor in self.models['subgraphs'][0]['tensors']:
                if tensor['name'] == buffer_name:
                    tensor['quantization'] = {'min': [quantization_param[buffer_name]['min']],
                                              'max': [quantization_param[buffer_name]['max']],
                                              'scale': [quantization_param[buffer_name]['scale']],
                                              'zero_point': [int(quantization_param[buffer_name]['zero_point'])]}
                    tensor['type'] = 'INT8'
        for op in self.models['subgraphs'][0]['operators']:
            opcode_index = op.get('opcode_index', 0)
            op_name = self.models['operator_codes'][opcode_index]['builtin_code']
            outputs = op['outputs']
            requirements = QUANTIZATIONREQUIREMENTS[op_name]
            for output_idx in range(len(outputs)):
                outputs_idx = 'Output_{}'.format(output_idx)
                requirement = requirements.get(outputs_idx)
                if not requirement:
                    continue
                if requirement['restriction'] == Restriction.SAME_WITH_INPUT:
                    input_tensor_idx = op['inputs'][0]
                    input_tensor = self.get_tensor_by_index(input_tensor_idx)
                    output_tensor_idx = outputs[output_idx]
                    output_tensor = self.get_tensor_by_index(output_tensor_idx)
                    output_tensor['quantization'] = input_tensor['quantization']
                if requirement['restriction'] == Restriction.SOFTMAX:
                    scale = 1.0 / 256.0
                    zero_point = -128
                    output_tensor_idx = outputs[output_idx]
                    output_tensor = self.get_tensor_by_index(output_tensor_idx)
                    output_tensor['quantization']['scale'] = [scale]
                    output_tensor['quantization']['zero_point'] = [zero_point]
                if requirement['restriction'] == Restriction.CONCAT:
                    self.models['operator_codes'].append(
                        {
                            "builtin_code": "QUANTIZE",
                            "version": 2
                        }
                    )
                    inputs = op['inputs']
                    min_of_input = 9e25
                    max_of_input = -9e25
                    for input_idx in inputs:
                        input_tensor = self.get_tensor_by_index(input_idx)
                        min_of_input = min(
                            min_of_input, input_tensor['quantization']['min'][0])
                        max_of_input = max(
                            max_of_input, input_tensor['quantization']['max'][0])
                    scale = (max_of_input - min_of_input) / 255
                    zero_point = np.round(127 - max_of_input / scale)

                    for input_idx in inputs:
                        input_tensor = self.get_tensor_by_index(input_idx)
                        buffer_data = {}
                        self.models['buffers'].append(buffer_data)
                        tensor_info = {
                            'shape': input_tensor['shape'],
                            'type': 'INT8',
                            'buffer': len(self.models['buffers']) - 1,
                            'name': input_tensor['name'] + '_requantized',
                            'quantization': {
                                'min': [np.around(min_of_input, 6).tolist()],
                                'max': [np.around(max_of_input, 6).tolist()],
                                'scale': [np.around(scale, 6).tolist()],
                                'zero_point': [int(zero_point.tolist())]
                            }
                        }
                        self.models['subgraphs'][0]['tensors'].append(
                            tensor_info)
                        quantize_op = {
                            'opcode_index': len(self.models['operator_codes']) - 1,
                            'inputs': [input_idx],
                            'outputs': [len(self.models['subgraphs'][0]['tensors']) - 1]
                        }
                        for operator in self.models['subgraphs'][0]['operators']:
                            inps = operator['inputs']
                            new_inps = []
                            for inp in inps:
                                if inp == input_idx:
                                    new_inps.append(quantize_op['outputs'][0])
                                else:
                                    new_inps.append(inp)
                            operator['inputs'] = new_inps
                        self.models['subgraphs'][0]['operators'].append(
                            quantize_op)
                    input_tensor_idx = op['inputs'][0]
                    input_tensor = self.get_tensor_by_index(input_tensor_idx)
                    output_tensor_idx = outputs[output_idx]
                    output_tensor = self.get_tensor_by_index(output_tensor_idx)
                    output_tensor['quantization'] = input_tensor['quantization']

    def calculate_min_max(self, data, requirement):
        quantized_dimension = None
        # Get min max
        if requirement['granularity'] == QuantizeGranularity.PER_TENSOR:
            min_value = data.min()
            max_value = data.max()
        if requirement['granularity'] == QuantizeGranularity.PER_AXIS_DIM0:
            if len(data.shape) == 4:
                min_value = np.min(data, axis=(1, 2, 3))
                max_value = np.max(data, axis=(1, 2, 3))
                quantized_dimension = 0
            else:
                min_value = np.min(data)
                max_value = np.max(data)
        if requirement['granularity'] == QuantizeGranularity.PER_AXIS_DIM3:
            if len(data.shape) == 4:
                min_value = np.min(data, axis=(0, 1, 2))
                max_value = np.max(data, axis=(0, 1, 2))
                quantized_dimension = 3
            else:
                min_value = np.min(data)
                max_value = np.max(data)
        return min_value, max_value, quantized_dimension

    def calculate_scale_and_zero_point(self, inputs, data, min_value, max_value, requirement, vrange):
        # Get scale and zero_point
        if requirement['restriction'] is None:
            scale = (np.maximum(np.abs(max_value), np.abs(min_value))
                     ) * 2 / (vrange[1] - vrange[0])
            zero_point = np.zeros_like(scale)
        if requirement['restriction'] == Restriction.ZERO_POINT_TO_ZERO:
            scale = (np.maximum(np.abs(max_value), np.abs(min_value))
                     ) * 2 / (vrange[1] - vrange[0])
            zero_point = np.zeros_like(scale)
        if requirement['restriction'] == Restriction.CONV_2D_BIAS:
            # scale = input0_scale * input1_scale[...]
            input0_tensor = self.get_tensor_by_index(inputs[0])
            input0_scale = input0_tensor['quantization']['scale']
            input1_tensor = self.get_tensor_by_index(inputs[1])
            input1_scale = input1_tensor['quantization']['scale']
            scale = np.array(input0_scale) * np.array(input1_scale)
            zero_point = np.zeros_like(scale)
        data_len = len(data.shape)
        scale_len = len(scale.shape)
        for _ in range(data_len - scale_len):
            if requirement['granularity'] == QuantizeGranularity.PER_AXIS_DIM0:
                scale = np.expand_dims(scale, -1)
                zero_point = np.expand_dims(zero_point, -1)
            if requirement['granularity'] == QuantizeGranularity.PER_AXIS_DIM3:
                scale = np.expand_dims(scale, 0)
                zero_point = np.expand_dims(zero_point, 0)
        return scale, zero_point

    def quantize_parameter(self):
        # Quantize parameters
        for op in self.models['subgraphs'][0]['operators']:
            opcode_index = op.get('opcode_index', 0)
            op_name = self.models['operator_codes'][opcode_index]['builtin_code']
            inputs = op['inputs']
            requirements = QUANTIZATIONREQUIREMENTS[op_name]
            for input_idx in range(len(inputs)):
                tensor_idx = inputs[input_idx]
                tensor_type = self.get_tensor_type_by_index(tensor_idx)
                input_idx = 'Input_{}'.format(input_idx)
                requirement = requirements.get(input_idx)
                if not requirement:
                    continue
                if requirements.get('op_version') is not None:
                    self.models['operator_codes'][opcode_index]['version'] = requirements.get(
                        'op_version')
                if tensor_type == 'Buffer':
                    continue
                vrange = requirement['range']
                data, tensor = self.get_tensor_and_data_by_index(tensor_idx)
                dtype = DTYPES.get(tensor['type'], np.float32)
                min_value, max_value, quantized_dimension = self.calculate_min_max(
                    data, requirement)
                scale, zero_point = self.calculate_scale_and_zero_point(
                    inputs, data, min_value, max_value, requirement, vrange)
                # quantize float32 parameter to int8
                scale += 1e-38
                data = np.round(data / scale + zero_point)
                if requirement['data_type'] == 'int8':
                    data = data.astype(np.int8).flatten().tobytes()
                if requirement['data_type'] == 'int32':
                    data = data.astype(np.int32).flatten().tobytes()
                data = [data[i] for i in range(len(data))]
                self.set_tensor_data_by_index(tensor_idx, data)
                tensor['type'] = requirement['data_type'].upper()
                tensor['quantization'] = {'min': np.around(min_value, 6).astype(dtype).flatten().tolist(),
                                          'max': np.around(max_value, 6).astype(dtype).flatten().tolist(),
                                          'scale': np.around(scale, 6).astype(dtype).flatten().tolist(),
                                          'zero_point': zero_point.astype(np.int32).flatten().tolist()}
                if quantized_dimension is not None:
                    tensor['quantization']['quantized_dimension'] = quantized_dimension

    def convert(self):
        check_opset(self.models)
        self.quantize_buffer()
        self.quantize_parameter()
        return self.models


class Calibrator(object):
    def __init__(self, tflite, dataset):
        """
        :param tflite (str): tflite model file
        :param dataset (numpy.ndarray): represent dataset
        """
        self.dataset = dataset
        self.interpreter = tf.lite.Interpreter(tflite)
        self.input_details = self.interpreter.get_input_details()
        if len(self.input_details) > 1:
            raise ValueError(
                "Currently, model more than 1 input is unsupported.")
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.resize_tensor_input(
            self.input_details[0]['index'], self.input_details[0]['shape'])
        self.interpreter.allocate_tensors()
        self.quantization_param = {}
        for output_detail in self.output_details:
            self.quantization_param[output_detail['name']] = {}

    def run(self):
        # Forward on represent dataset to collect max and min value of each buffer
        alpha = 0.1
        for input_data in self.dataset:
            input_data = np.array(input_data).astype(np.float32)
            input_data = np.expand_dims(input_data, 0)
            input_shape = self.input_details[0]['shape']
            if not (input_shape == input_data.shape).all():
                input_data = np.transpose(input_data, (0, 2, 3, 1))
            self.interpreter.set_tensor(
                self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            for output_detail in self.output_details:
                output_tensor = self.interpreter.get_tensor(
                    output_detail['index'])
                if not self.quantization_param[output_detail['name']].get('max'):
                    self.quantization_param[output_detail['name']
                                            ]['max'] = output_tensor.max()
                # Using exponential moving averages to smooth max value and min value
                # Reference： https://arxiv.org/pdf/1712.05877.pdf
                if output_tensor.max() > self.quantization_param[output_detail['name']]['max']:
                    self.quantization_param[output_detail['name']]['max'] = alpha * output_tensor.max(
                    ) + (1 - alpha) * self.quantization_param[output_detail['name']]['max']

                if not self.quantization_param[output_detail['name']].get('min'):
                    self.quantization_param[output_detail['name']
                                            ]['min'] = output_tensor.min()
                if output_tensor.min() < self.quantization_param[output_detail['name']]['min']:
                    self.quantization_param[output_detail['name']]['min'] = alpha * output_tensor.min(
                    ) + (1 - alpha) * self.quantization_param[output_detail['name']]['min']

        for name in self.quantization_param:
            max_v = self.quantization_param[name]['max']
            min_v = self.quantization_param[name]['min']
            scale = (max_v - min_v) / 255
            zero_point = np.round(127 - max_v / scale)
            self.quantization_param[name]['max'] = np.around(max_v, 6).tolist()
            self.quantization_param[name]['min'] = np.around(min_v, 6).tolist()
            self.quantization_param[name]['scale'] = np.around(
                scale, 6).tolist()
            self.quantization_param[name]['zero_point'] = zero_point.tolist()
        return self.quantization_param
