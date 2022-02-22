# Copyright 2018,2019,2020,2021 Sony Corporation.
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
import collections
from functools import partial

import nnabla.utils.converter
import numpy as np


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
    return ret.astype(np.float32)


def create_nnabart_info(nnp, batch_size):
    class info:
        pass
    executor = nnabla.utils.converter.select_executor(nnp)
    # Search network.
    network = nnabla.utils.converter.search_network(
        nnp, executor.network_name)

    if network is None:
        print('Network for executor [{}] is not found.'.format(
            executor.network_name))
        return
    print('Using network [{}].'.format(executor.network_name))

    info._batch_size = batch_size
    if batch_size < 0:
        info._batch_size = network.batch_size

    info._network_name = executor.network_name

    parameters = collections.OrderedDict()
    for p in nnp.protobuf.parameter:
        parameters[p.variable_name] = p

    variables = collections.OrderedDict()
    for v in network.variable:
        variables[v.name] = v

    info._generator_variables = {}
    info._num_of_gen_variables = len(executor.generator_variable)
    for v in executor.generator_variable:
        v_info = variables[v.variable_name]
        shape = [d if d > 0 else info._batch_size for d in v_info.shape.dim]
        data = generate_value(v.type, shape, v.multiplier)
        info._generator_variables[v.variable_name] = data

    info._input_variables = []
    info._num_of_inputs = len(executor.data_variable)
    info._input_buffer_sizes = []
    for n, i in enumerate(executor.data_variable):
        info._input_variables.append(i.variable_name)
        v = variables[i.variable_name]
        info._input_buffer_sizes.append(
            nnabla.utils.converter.calc_shape_size(v.shape, info._batch_size))

    info._output_variables = []
    info._num_of_outputs = len(executor.output_variable)
    info._output_buffer_sizes = []
    for n, o in enumerate(executor.output_variable):
        info._output_variables.append(o.variable_name)
        v = variables[o.variable_name]
        info._output_buffer_sizes.append(
            nnabla.utils.converter.calc_shape_size(v.shape, info._batch_size))

    info._param_variables = []
    info._num_of_params = len(executor.parameter_variable)
    for n, p in enumerate(executor.parameter_variable):
        info._param_variables.append(p.variable_name)

    # Prepare variable buffers
    info._variable_sizes = []
    info._variable_buffer_index = collections.OrderedDict()
    info._variable_buffer_size = collections.OrderedDict()

    info._buffer_ids = {}
    buffer_index = 0
    for n, v in enumerate(network.variable):
        size = nnabla.utils.converter.calc_shape_size(
            v.shape, info._batch_size)
        info._variable_sizes.append(size)
        if v.type == 'Buffer':
            info._variable_buffer_index[buffer_index] = [n]
            for vid in info._variable_buffer_index[buffer_index]:
                info._buffer_ids[vid] = buffer_index

            if buffer_index in info._variable_buffer_size:
                if size > info._variable_buffer_size[buffer_index]:
                    info._variable_buffer_size[buffer_index] = size
            else:
                info._variable_buffer_size[buffer_index] = size
            buffer_index += 1

    info._parameters = parameters
    info._variables = variables
    info._network = network
    info._function_info = nnabla.utils.converter.get_function_info()
    info._convert_context = {}
    return info


def revise_buffer_size(info, settings):
    '''
    This function is used to revise buffer size, use byte
    as its unit, instead of data item.
    This is only used for nnb, not for csrc.
    When settings contains user customized data type, not pure
    FLOAT32, it affects the memory consumption.
    '''
    size_mapping = {
        'FLOAT32': 4,
        'FIXED16': 2,
        'FIXED8': 1
    }

    var_dict = settings['variables']
    buffer_index = 0
    info._variable_sizes = []
    info._variable_buffer_index = collections.OrderedDict()
    info._variable_buffer_size = collections.OrderedDict()
    info._buffer_ids = {}
    for n, v in enumerate(info._network.variable):
        byte_per_item = size_mapping.get(var_dict.get(
            v.name, 'FLOAT32').split('_')[0], 4)
        size = nnabla.utils.converter.calc_shape_size(
            v.shape, info._batch_size) * byte_per_item
        info._variable_sizes.append(size)
        if v.type == 'Buffer':
            info._variable_buffer_index[buffer_index] = [n]
            for vid in info._variable_buffer_index[buffer_index]:
                info._buffer_ids[vid] = buffer_index

            info._variable_buffer_size[buffer_index] = size
            buffer_index += 1


def affine_transpose_weight(params, info, func):
    if 'Affine' in info._convert_context:
        transposed = info._convert_context['Affine']
    else:
        transposed = set()

    for idx in params:
        weight_name = func.input[idx]
        if weight_name in transposed:
            return
        w_shape = info._variables[weight_name].shape.dim[:]
        if weight_name in info._parameters:
            w_data = info._parameters[weight_name]
            transposed.add(weight_name)
            info._convert_context['Affine'] = transposed
        else:
            print(
                "WARNING: affine weight is not transposed. Since it is not included in .nntxt/.nnp")
        i_num = w_shape[0]
        data = np.array(w_data.data[:])
        data = data.reshape(int(i_num), -1)
        data = np.transpose(data)
        del info._parameters[weight_name].data[:]
        info._parameters[weight_name].data.extend(data.flatten())


def pack_bin_conv_unused_weight(index, info, func):
    weight_name = func.input[index]
    d = info._parameters[weight_name].data[:]
    d = d[0:1]  # TRUNC TO 1
    del info._parameters[weight_name].data[:]
    info._parameters[weight_name].data.extend(d)


NNB_PREPROCESS_LIST = {
    'Affine': partial(affine_transpose_weight, [1]),
    'BinaryConnectAffine': partial(affine_transpose_weight, [1, 2]),
    'BinaryWeightAffine': partial(affine_transpose_weight, [1, 2]),
    'BinaryWeightConvolution': partial(pack_bin_conv_unused_weight, 1),
    'BinaryConnectConvolution': partial(pack_bin_conv_unused_weight, 1)
}

CSRC_PREPROCESS_LIST = {
    'Affine': partial(affine_transpose_weight, [1]),
    'BinaryConnectAffine': partial(affine_transpose_weight, [1, 2]),
    'BinaryWeightAffine': partial(affine_transpose_weight, [1, 2])
}

PREPROCESS_DICT = {
    'CSRC': CSRC_PREPROCESS_LIST,
    'NNB': NNB_PREPROCESS_LIST
}


def preprocess_for_exporter(info, exporter_name):
    if exporter_name in PREPROCESS_DICT:
        preprocess_list = PREPROCESS_DICT[exporter_name]
    else:
        return

    for func in info._network.function:
        if func.type in preprocess_list:
            preprocessor = preprocess_list[func.type]
            if callable(preprocessor):
                preprocessor(info, func)
