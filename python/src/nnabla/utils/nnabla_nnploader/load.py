import os
import google.protobuf.text_format as text_format
import tempfile
import zipfile
import shutil

from collections import defaultdict

import nnabla as nn
import nnabla.function as F
from nnabla.utils import nnabla_pb2
from nnabla.parameter import get_parameter
from nnabla.utils.load_function import _create_function_instance


def _load_nnp_to_proto(nnp_path):
    proto = nnabla_pb2.NNablaProtoBuf()

    tmpdir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(nnp_path, "r") as nnp:
            for name in nnp.namelist():
                _, ext = os.path.splitext(name)
                if name == "nnp_version.txt":
                    pass  # Currently nnp_version.txt is ignored
                elif ext in [".nntxt", ".prototxt"]:
                    nnp.extract(name, tmpdir)
                    with open(os.path.join(tmpdir, name), "rt") as f:
                        text_format.Merge(f.read(), proto)
                elif ext in [".protobuf", ".h5"]:
                    nnp.extract(name, tmpdir)
                    nn.load_parameters(os.path.join(tmpdir, name))
    finally:
        shutil.rmtree(tmpdir)

    return proto


def _create_function(ctx, funtion_proto, batch_size):
    # todo: arrange weight name for NNC

    if funtion_proto.type == "Reshape":  # if batch_size = -1, something wrong?
        reshape_shape = (batch_size,) + \
                        tuple(funtion_proto.reshape_param.shape.dim)
        function_instance = F.Reshape(ctx, shape=reshape_shape)
    elif funtion_proto.type == "RepeatStart":
        function_instance = F.Identity(ctx)
    elif funtion_proto.type == "RepeatEnd":
        function_instance = F.Identity(ctx)
    elif funtion_proto.type == "RecurrentOutput":
        function_instance = F.Stack(ctx, axis=funtion_proto.recurrent_param.axis)
    elif funtion_proto.type == "RecurrentInput":
        function_instance = F.Split(ctx, axis=funtion_proto.recurrent_param.axis)
    elif funtion_proto.type == "Delay":
        function_instance = F.Identity(ctx)
    else:
        function_instance = _create_function_instance(ctx, funtion_proto)

    return function_instance


def _create_input_variables(function_proto, variable_proto_dict, batch_size):
    inputs = list()
    no_param_variables = list()
    for variable_name in function_proto.input:
        shape = tuple([d if d > 0 else batch_size
                       for d in variable_proto_dict[variable_name].shape.dim])

        param = get_parameter(variable_name)

        if param is None:  # this means variable is not a parameter but input data of this function
            param = nn.Variable(shape)
            no_param_variables.append((variable_name, param))
        else:
            assert param.shape == shape

        inputs.append(param)

    return inputs, no_param_variables


def _create_nnabla_graph(proto, ctx=None):
    if ctx is None:
        ctx = nn.get_current_context()

    # setup all functions
    all_variables = dict()
    for network_proto in proto.network:
        all_variables[network_proto.name] = dict()
        variable_connections = defaultdict(list)
        variable_proto_dict = {v.name: v for v in network_proto.variable}

        for function_proto in network_proto.function:
            function_instance = _create_function(ctx, function_proto, network_proto.batch_size)

            inputs, no_param_variables = _create_input_variables(function_proto,
                                                                 variable_proto_dict, network_proto.batch_size)

            outputs = function_instance(*inputs)

            outputs = outputs if isinstance(outputs, list) else [outputs, ]

            no_param_variables += [tuple(x) for x in zip(function_proto.output, outputs)]

            for var_name, var in no_param_variables:
                variable_connections[var_name].append(var)

        # connect redundant variables
        for key, vs in variable_connections.items():
            if len(vs) == 1:
                variable = vs[0]
            else:
                # todo: merge variables (call connect function like connect(*vs) for all combination)
                # todo: change need_grad attr. False to True?
                variable = vs[0]  # dummy, replace it to "variable = connect(*vs)"

            all_variables[network_proto.name][key] = variable

        import ipdb; ipdb.set_trace()

    return all_variables


def _create_executors(proto, all_variables):
    executors = dict()

    class Executor:
        inputs = dict()
        outputs = dict()

    for executor_proto in proto.executor:
        executor = Executor()
        executor.name = executor_proto.name
        executor.network_name = executor_proto.network_name

        # get inputs
        for input_proto in executor_proto.data_variable:
            variable_name = input_proto.variable_name

            executor.inputs[variable_name] = all_variables[executor.network_name][variable_name]

        # get outputs
        for output_proto in executor_proto.output_variable:
            variable_name = output_proto.variable_name

            executor.outputs[variable_name] = all_variables[executor.network_name][variable_name]

        executors[executor_proto.name] = executor

    return executors


def load_nnp_for_nnabla(filepath):
    """
    load_nnp_for_nnabla
        Load network information from nnp files.

    Args:
        filenames: List of filenames.

    Returns:
        executors: List of Executor class which includes information about inputs, outputs and network name.

    usage:
        Executors = load_nnp_for_nnabla("/path/to/hoge.nnp")
        graph_inputs = Executors["target_executor_name"].inputs
        graph_outputs = Executors["target_executor_name"].outputs

        # if you want to get network name looked up from the specific executor, you can do this like below
        target_network_name = Executors["target_executor_name"].network_name

        for variable_name, variable in graph_inputs:
            variable.d = (data with real value for variable_name, say numpy.array)

        out = F.sink(*graph_outputs.values())

        out.forward() # you can call any function which can be used for nnabla graph
    """

    _, ext = os.path.splitext(filepath)

    if ext == ".nnp":
        proto = _load_nnp_to_proto(filepath)

        if hasattr(proto, "network"):
            all_variables = _create_nnabla_graph(proto, ctx=None)  # Currently ctx is from nn.get_default_context()
        else:
            raise AssertionError(".nntxt dose not include network information")

        if hasattr(proto, "executor"):
            executors = _create_executors(proto, all_variables)
        else:
            # todo : Currently if nnp dose not include executor information, raise Error
            raise AssertionError(".nntxt dose not include network information")

    else:
        raise NotImplementedError("Currently extension of file for loading must be ['.nnp', ]")

    import ipdb; ipdb.set_trace()

    return executors
