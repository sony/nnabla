import collections
import numpy as np
from functools import partial

import nnabla.utils.converter


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
        print('Network for executor [{}] does not found.'.format(
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


NNB_PREPROCESS_LIST = {
    'Affine': partial(affine_transpose_weight, [1]),
    'BinaryConnectAffine': partial(affine_transpose_weight, [1, 2]),
    'BinaryWeightAffine': partial(affine_transpose_weight, [1, 2])
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
