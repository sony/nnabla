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

import struct
from collections import OrderedDict
import nnabla.logger as logger
from nnabla.utils import nnabla_pb2
from onnx import (ModelProto, TensorProto, AttributeProto)

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
    "Relu": "ReLU",
    "Concat": "Concatenate",
    "Conv": "Convolution",
    "GlobalAveragePool": "GlobalAveragePooling",
    "MaxPool": "MaxPooling",
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


def convert_to_function(node, base_name, func_counter):
    """Convert given node to corresponding function"""
    func = nnabla_pb2.Function()
    func.type = onnx_optype_to_nnabla_function_type.get(node.op_type, node.op_type)
    # NNabla requires each function to have a unique name.
    # If the node's name already has something set,
    # we are going to use it.
    # If no name is set (which is allowed in ONNX) we
    # are going to generate a name
    if node.name:
        func.name = node.name
    else:
        # Node name was not specified.
        # We are going to generate a name by counting
        # how many times a function was used.
        # (or we might use some kind of random number and hash it)
        count = 0
        if func.type in func_counter:
            # This function has been used already.
            # Get the current count
            count = func_counter[func.type]
        func.name = "{}/{}_{}".format(base_name, func.type, count)
        # Store the current usage count
        func_counter[func.type] = count+1
    func.input.extend(node.input)
    func.output.extend(node.output)
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
    elif node.op_type == "Softmax":
        logger.warning(SOFTMAX_WARNING)
        # default to channel axis
        func.softmax_param.axis = DEFAULT_SOFTMAX_AXIS
        for attr in node.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Softmax axis must be a single integer")
                func.softmax_param.axis = attr.i
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
            # We do not set 'kernel_shape' to NNabla
            # since NNabla doesn't have a parameter for it
            # (it will be inferred from weight input)
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
        # NNabla requires for the dimensions of strides, pads, dilations to match.
        # We align the dimensions for all three attributes to the shortest one
        dim = min(dims)
        if strides:
            cp.stride.dim.extend(strides[:dim])
        if pads:
            cp.pad.dim.extend(pads[:dim])
        if dilations:
            cp.dilation.dim.extend(dilations[:dim])
        else:
            # Set default values.
            # Do we really need this? (Default value should be set by NNabla)
            cp.dilation.dim.extend([1 for _ in range(dim)])
    elif node.op_type == "MaxPool":
        mpp = func.max_pooling_param
        dims = []
        strides = []
        pads = []
        kernel = []
        for attr in node.attribute:
            if attr.name == "strides":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for strides in MaxPool op_type")
                strides.extend(attr.ints)
                dims.append(len(strides))
            elif attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for pads in MaxPool op_type")
                pads.extend(attr.ints)
                dims.append(len(pads))
            elif attr.name == "kernel_shape":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for kernel_shape in MaxPool op_type")
                kernel.extend(attr.ints)
                dims.append(len(kernel))
        # NNabla requires for the dimensions of strides, pads, kernels to match.
        # We align the dimensions for all three attributes to the shortest one
        dim = min(dims)
        if strides:
            mpp.stride.dim.extend(strides[:dim])
        if pads:
            mpp.pad.dim.extend(pads[:dim])
        if kernel:
            mpp.kernel.dim.extend(kernel[:dim])
        # Always ignore borders in order to match ONNX(caffe2) results?
        # Not quite sure yet.
        mpp.ignore_border = True

    return func


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
        f = convert_to_function(n, graph.name, func_counter)
        # Gather all unique names for input and output
        for i in f.input:
            all_vars[i] = None
        for o in f.output:
            all_vars[o] = None
        network.function.extend([f])

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
        num = len(init.raw_data) // 4  # four bytes per float
        # logger.log(99, "raw_data num: {}".format(num))
        data = struct.unpack(str(num)+'f', init.raw_data)
        p.data.extend(data)
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
        del all_vars[v.name]
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
