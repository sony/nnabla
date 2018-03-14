# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

from collections import OrderedDict
import nnabla.logger as logger
from nnabla.utils import nnabla_pb2
from onnx import (ModelProto, TensorProto, AttributeProto)
import numpy as np

from .utils import *

# We default to concat the channel axis
# so the concat results match with caffe2
DEFAULT_CONCAT_AXIS = 1
DEFAULT_SOFTMAX_AXIS = 1
# ONNX does not have the concept of executors.
# We will add a single executor to NNP when converted from ONNX,
# and set a default name to it.
DEFAULT_EXECUTOR_NAME = "exec_0"

# Dictionary used to convert ONNX op_type to NNabla function names
onnx_optype_to_nnabla_function_type = {
    # optype with same names
    "Dropout": "Dropout",
    "Softmax": "Softmax",
    "BatchNormalization": "BatchNormalization",
    # optype with different names
    "Relu": "ReLU",
    "Concat": "Concatenate",
    "Conv": "Convolution",
    "GlobalAveragePool": "GlobalAveragePooling",
    "MaxPool": "MaxPooling",
    "AveragePool": "AveragePooling",
    "Sum": "Add2",
    "Gemm": "Affine"
}


def onnx_value_info_proto_to_variable(info, network):
    if not info.type.HasField("tensor_type"):  # accepting only tensor
        raise ValueError("Only TensorProto is allowed as ValueInfoProto's type for info.name (Got {})"
                         .format(info.name, info.type))
    t = info.type.tensor_type
    v = network.variable.add()
    v.name = info.name
    v.shape.dim.extend([x.dim_value for x in t.shape.dim])
    return v

def set_kernel_parameter(node, kp):
    """Set kernel related parameters(strides, pads, kernel_shape) to the given parameter"""
    dims = []
    strides = []
    pads = []
    kernel = []
    for attr in node.attribute:
        if attr.name == "strides":
            if attr.type != AttributeProto.INTS:
                raise ValueError("Only INTS are supported for strides in {}"
                                 .format(node.op_type))
            strides.extend(attr.ints)
            dims.append(len(strides))
        elif attr.name == "pads":
            if attr.type != AttributeProto.INTS:
                raise ValueError("Only INTS are supported for pads in {}"
                                 .format(node.op_type))
            pads.extend(attr.ints)
            dims.append(len(pads))
        elif attr.name == "kernel_shape":
            if attr.type != AttributeProto.INTS:
                raise ValueError("Only INTS are supported for kernel_shape in {}"
                                 .format(node.op_type))
            kernel.extend(attr.ints)
            dims.append(len(kernel))
        else:
            raise ValueError("Unsupported attribute {} was specified at {}"
                             .format(attr.name, node.op_type))
    # NNabla requires for the dimensions of strides, pads, kernels to match.
    # We align the dimensions for all three attributes to the shortest one
    dim = min(dims)
    if strides:
        kp.stride.dim.extend(strides[:dim])
    if pads:
        kp.pad.dim.extend(pads[:dim])
    else:
        # In case we don't have padding set,
        # we set zero padding just in case NNabla does not set the
        # default padding values correctly (such as in AveragePooling).
        # This code should not be needed if NNabla handles default values correctly.
        kp.pad.dim.extend([0]*dim)
    if kernel:
        kp.kernel.dim.extend(kernel[:dim])


def update_function_counter(func_type, func_counter, count):
    # Store the current usage count
    func_counter[func_type] = count+1


def generate_function_name(func_type, base_name, func_counter):
    # We are going to generate a name by counting
    # how many times a function was used.
    # (or we might use some kind of random number and hash it)
    count = 0
    if func_type in func_counter:
        # This function has been used already.
        # Get the current count
        count = func_counter[func_type]
    return "{}/{}_{}".format(base_name, func_type, count), count

def set_function_name(func, nodeName, base_name, func_counter):
    """Set a sufficient name for the function"""
    # NNabla requires each function to have a unique name.
    # If the node's name already has something set,
    # we are going to use it.
    # If no name is set (which is allowed in ONNX) we
    # are going to generate a name
    if nodeName:
        func.name = nodeName
    else:
        # Node name was not specified.
        func.name, count = generate_function_name(func.type, base_name, func_counter)
        update_function_counter(func.type, func_counter, count)


def generate_default_function(node, base_name, func_counter):
    """Generate a default function from the given node
    """
    ft = onnx_optype_to_nnabla_function_type.get(node.op_type)
    if ft is None:
        raise ValueError("op_type {} is currently not supported for NNP conversion".format(node.op_type))
    func = nnabla_pb2.Function()
    func.type = ft
    set_function_name(func, node.name, base_name, func_counter)
    func.input.extend(node.input)
    func.output.extend(node.output)
    return func

def generate_transpose(node_name, in_name, out_name, base_name, func_counter):
    """Generate a Transpose operator to transpose the specified buffer.
    We assume the buffer is two dimensional.
    """
    trans = nnabla_pb2.Function()
    trans.type = "Transpose"
    set_function_name(trans, node_name, base_name, func_counter)
    trans.input.extend([in_name])
    trans.output.extend([out_name])
    tp = trans.transpose_param
    tp.axes.extend([1, 0])  # switch H and W
    return trans

def convert_to_functions(node, base_name, initializers, func_counter):
    """Convert given node to corresponding functions.
    A node is usually converted to a single function,
    but some nodes end up as a composition of functions.
    """
    func_list = []
    func = generate_default_function(node, base_name, func_counter)
    if node.op_type == "Concat":
        # Since concat axis is currently not required in ONNX,
        # the default axis depends on which backend we use.
        # For now we are comparing with caffe2, so we are
        # defaulting to the channel axis if the axis is not specified.
        # https://github.com/onnx/onnx/issues/374
        func.concatenate_param.axis = DEFAULT_CONCAT_AXIS
        for attr in node.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Axis type must be a single integer")
                # The axis was specified so we use it
                func.concatenate_param.axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
    elif node.op_type == "Softmax":
        logger.warning(SOFTMAX_WARNING)
        # default to channel axis
        func.softmax_param.axis = DEFAULT_SOFTMAX_AXIS
        for attr in node.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Softmax axis must be a single integer")
                func.softmax_param.axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
    elif node.op_type == "Dropout":
        # Dropout requires a ratio to be set
        for attr in node.attribute:
            if attr.name == "is_test":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Dropout is_test must be a single integer")
                if attr.i != 0:
                    # is_test is True meaning we will not be applying dropout.
                    # We are simply going to pass through the input values
                    # by using the Identity function
                    func.ClearField("dropout_param")
                    func.type = "Identity"

                    if len(node.output) > 1:
                        # Identity only allows a single output,
                        # while a dropout node may have two outputs (result + mask)
                        # We will drop the mask output (which should be the second one)
                        del func.output[:]
                        func.output.extend([node.output[0]])
                    # We break here so we don't write any needless attributes
                    break
            elif attr.name == "ratio":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Dropout ratio must be a single float")
                func.dropout_param.p = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
    elif node.op_type == "Conv":
        cp = func.convolution_param
        # We shouldn't need these default settings
        # since NNabla will set these for us
        cp.base_axis = 1
        cp.group = 1
        dims = []
        pads = []
        strides = []
        dilations = []
        for attr in node.attribute:
            if attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for pads in Conv op_type")
                pads.extend(attr.ints)
                dims.append(len(pads))
            elif attr.name == "strides":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for strides in Conv op_type")
                strides.extend(attr.ints)
                dims.append(len(strides))
            elif attr.name == "dilations":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for dilations in Conv op_type")
                dilations.extend(attr.ints)
                dims.append(len(dilations))
            elif attr.name == "group":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for group in Conv op_type")
                cp.group = attr.int
            elif attr.name == "kernel_shape":
                # We do not set 'kernel_shape' to NNabla
                # since NNabla doesn't have a parameter for it
                # (it will be inferred from weight input)
                pass
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        # NNabla requires for the dimensions of strides, pads, dilations to match.
        # We align the dimensions for all three attributes to the shortest one
        dim = min(dims)
        if strides:
            cp.stride.dim.extend(strides[:dim])
        if pads:
            cp.pad.dim.extend(pads[:dim])
        else:
            # Set default values.
            # Do we really need this? (Default value should be set by NNabla)
            cp.pad.dim.extend([0 for _ in range(dim)])
        if dilations:
            cp.dilation.dim.extend(dilations[:dim])
        else:
            # Set default values.
            # Do we really need this? (Default value should be set by NNabla)
            cp.dilation.dim.extend([1 for _ in range(dim)])
    elif node.op_type == "MaxPool":
        mpp = func.max_pooling_param
        set_kernel_parameter(node, mpp)
        # Always ignore borders in order to match ONNX(caffe2) results?
        # Not quite sure yet.
        mpp.ignore_border = True
    elif node.op_type == "AveragePool":
        app = func.average_pooling_param
        set_kernel_parameter(node, app)
        # Always ignore borders in order to match ONNX(caffe2) results?
        # Not quite sure yet.
        app.ignore_border = True
    elif node.op_type == "BatchNormalization":
        # We need to rearrange the input data order.
        # ONNX BatchNormalization input order: X, scale, bias, mean, variance
        # NNabla BatchNormalization input order: X, beta, gamma, mean, variance
        nnp_order = [0, 2, 1, 3, 4]
        if len(node.input) != len(nnp_order):
            raise ValueError("The number of BatchNormalization input must be {}".format(len(nnp_order)))
        nnp_input = [node.input[i] for i in nnp_order]
        del func.input[:]
        func.input.extend(nnp_input)
        bnp = func.batch_normalization_param
        # Set default axis.
        # We shouldn't need this if the default is set properly
        bnp.axes.extend([1])
        for attr in node.attribute:
            if attr.name == "is_test":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for is_test in BatchNormalization op_type")
                if attr.i == 0:
                    raise ValueError("BatchNormalization with is_test=False is currently not supported")
                is_test = (attr.i == 1)
                bnp.batch_stat = not is_test
            elif attr.name == "epsilon":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Only FLOAT is supported for epsilon in BatchNormalization op_type")
                bnp.eps = attr.f
            elif attr.name == "momentum":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Only FLOAT is supported for momentum in BatchNormalization op_type")
                bnp.decay_rate = attr.f
            elif attr.name == "consumed_inputs":
                # Caffe2 currently uses an undocumented attribute consumed_inputs for BatchNormalization.
                # Since NNabla does not need this, we ignore it
                pass
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
    elif node.op_type == "Gemm":
        ap = func.affine_param
        ap.base_axis = 1
        transposed_postfix = "_transposed"
        input = node.input[:]
        # Check the dimension of all inputs
        x = input[0]
        weight = input[1]
        bias = input[2]
        for init in initializers:
            if init.name == x or init.name == weight:
                # must be two dimensional
                if len(init.dims) != 2:
                    raise ValueError("Only two dimensional input is currently supported for Gemm input tensor A and B ({})".format(init.name))
            elif init.name == bias:
                # Must be one dimensional
                if len(init.dims) != 1:
                    raise ValueError("Only one dimensional input is currently supported for Gemm input tensor C ({})".format(init.name))

        for attr in node.attribute:
            if attr.name == "transA":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for transA in {} op_type".format(node.op_type))
                # We need to transpose the input weight beforehand
                # since NNabla does not support transpose with Affine.
                # Add a new intermediate buffer for transposition,
                # and rewire the buffer as input.
                ain = node.input[0]
                aout = ain+transposed_postfix
                transA = generate_transpose(node.name, ain, aout, base_name, func_counter)
                func_list.append(transA)
                input[0] = aout  # rewire input to transposed input
            elif attr.name == "transB":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for transB in {} op_type".format(node.op_type))
                # same as transA
                bin = node.input[1]
                bout = bin+transposed_postfix
                transB = generate_transpose(node.name, bin, bout, base_name, func_counter)
                func_list.append(transB)
                input[1] = bout  # rewire input to transposed input
            elif attr.name == "broadcast":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for broadcast in {} op_type".format(node.op_type))
                # Affine broadcasts Bias vector automatically if the bias is one dimension
                # so we don't have to do anything here
                pass
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        # update Gemm input with the converted inputs
        del func.input[:]
        func.input.extend(input)
    func_list.append(func)
    return func_list

def convert_parameter_shape(pb):
    """Convert the shape of some parameters so they fit NNabla's requirements.
    We do this as a post conversion because in the future we may be able to
    delete the whole conversion if NNabla's code gets changed"""
    if len(pb.network) != 1:
        raise ValueError("NNP with more then a single network is currently not supported")
    net = pb.network[0]
    batch_norm_constants = []
    for f in net.function:
        if f.type == "BatchNormalization":
            # BatchNormalization in ONNX requires the scale, bias, mean, and variance input to be
            # one dimensional (https://github.com/onnx/onnx/blob/master/docs/Operators.md#batchnormalization).
            # However in NNabla these input must have a specific shape that matches the input shape.
            # For example if the input shape is (1,3,3,3), the above variables must have the shape (1,3,1,1) and not (3).
            # (1,3,1,1) is actually the same as a one-dimensional tensor of size 3,
            # but NNabla's check currently does not allow this.
            # Thus, we convert the shape of the above input so we can pass NNabla's check.
            # If NNabla lightens the shape check, we should be able to remove this conversion.
            batch_norm_constants.extend(f.input[1:5])  # copy all input names for scale, bias, mean, variance

    # This loop should be fairly slow since we loop through all variables and parameters per constant
    for c in batch_norm_constants:
        # Reshape all BatchNormalization constant inputs assuming the size is (1,size,1,1)
        for v in net.variable:
            if v.name == c:
                size = v.shape.dim[0]
                del v.shape.dim[:]
                v.shape.dim.extend([1, size, 1, 1])
                break
        for p in pb.parameter:
            if p.variable_name == c:
                size = p.shape.dim[0]
                del p.shape.dim[:]
                p.shape.dim.extend([1, size, 1, 1])
                break




def onnx_graph_to_nnp_protobuf(pb, graph):
    network = pb.network.add()
    network.name = graph.name

    # We use an OrderedDict and not a set
    # to preserve order
    all_vars = OrderedDict()
    func_counter = {}
    # convert nodes
    for n in graph.node:
        # We do not allow any operator from an unknown domain
        if not (n.domain == '' or n.domain == NNABLA_DOMAIN):
            raise ValueError("Unsupported operator from domain {} was found".format(n.domain))
        fl = convert_to_functions(n, graph.name, graph.initializer, func_counter)
        # Gather all unique names for input and output
        for f in fl:
            for i in f.input:
                all_vars[i] = None
            for o in f.output:
                all_vars[o] = None
        network.function.extend(fl)

    # convert parameters
    # We use an OrderedDict and not a set
    # to preserve order
    param_vars = OrderedDict()
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            raise ValueError("Only floating point data is supported for parameters {} (Got {})"
                             .format(init.name, init.data_type))
        p = pb.parameter.add()
        p.variable_name = init.name
        p.shape.dim.extend(init.dims)
        # convert raw bytestream to floating points
        #num = len(init.raw_data) // 4  # four bytes per float
        # logger.log(99, "raw_data num: {}".format(num))
        p.data.extend(np.fromstring(init.raw_data, dtype=np.float32))
        p.need_grad = False
        # Keep the list of all initializer names
        param_vars[init.name] = None
    # We need to distinguish constant parameters (which become 'Parameter' in NNabla)
    # from input/output variables (which become 'Buffer' in NNabla).
    # Constant parameters appear in the initializer list so we keep
    # all names of variables from the initializer and compare them with
    # the names we gathered in all_vars.
    # The names that only appear in all_vars are the input/output variables.

    # convert Input/Output ValueInfoProto
    # to Variable
    in_list = []
    param_list = []
    out_list = []
    var_type_buffer = "Buffer"
    for i in graph.input:
        v = onnx_value_info_proto_to_variable(i, network)
        if v.name in param_vars:
            # This input is a parameter
            v.type = "Parameter"
            v.initializer.type = "Constant"
            v.initializer.multiplier = 1.0
            param_list.append(v)
        else:
            # This input is a variable
            v.type = var_type_buffer
            in_list.append(v)
        if v.name in all_vars:
            del all_vars[v.name]
        else:
            # We come here when a buffer (usually a parameter) is included in
            # graph.input and graph.initializer,
            # but was not actually used as input for any node.
            # No one is using this buffer so we show a warning and ignore it.
            logger.warning("Input buffer {} is not used as input for any node.".format(v.name))
    for o in graph.output:
        v = onnx_value_info_proto_to_variable(o, network)
        v.type = var_type_buffer
        out_list.append(v)
        del all_vars[v.name]

    for varg in all_vars:
        # We add all remaining variables as intermediate buffer
        # We leave the buffer size of all intermediate buffers empty
        v = network.variable.add()
        v.type = var_type_buffer
        v.name = varg

    # Add executor for target network
    exe = pb.executor.add()
    exe.name = DEFAULT_EXECUTOR_NAME
    exe.network_name = network.name
    for iv in in_list:
        dv = exe.data_variable.add()
        dv.variable_name = iv.name
        dv.data_name = iv.name
    for ov in out_list:
        outv = exe.output_variable.add()
        outv.variable_name = ov.name
        outv.data_name = ov.name
    for pv in param_list:
        p = exe.parameter_variable.add()
        p.variable_name = pv.name
    convert_parameter_shape(pb)

def onnx_model_to_nnp_protobuf(model):
    pb = nnabla_pb2.NNablaProtoBuf()
    if model.ir_version < MIN_ONNX_IR_VERSION:
        raise ValueError("Older ONNX IR versions are currently not supported")
    for opset in model.opset_import:
        if opset.domain == "":
            # ONNX opset.
            # Check if we have the correct version
            if opset.version < MIN_ONNX_OPSET_VERSION:
                raise ValueError("Older ONNX opsets are currently not supported")
        else:
            raise ValueError("Unsupported opset from domain {}".format(opset.domain))

    # convert onnx model to nnabla protobuf
    # logger.log(99, "Converting ONNX made by {}.".format(model.producer_name))

    # convert graph
    onnx_graph_to_nnp_protobuf(pb, model.graph)

    class nnp:
        pass
    nnp.protobuf = pb
    nnp.other_files = []
    return nnp


class OnnxReader:
    def __init__(self, file_path):
        self._file_path = file_path

    def read(self):
        model_proto = ModelProto()
        with open(self._file_path, "rb") as f:
            model_proto.ParseFromString(f.read())
        return onnx_model_to_nnp_protobuf(model_proto)
