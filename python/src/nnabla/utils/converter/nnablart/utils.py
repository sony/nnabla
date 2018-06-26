import collections

import nnabla.utils.converter


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
    return info
