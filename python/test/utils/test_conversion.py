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

import os
import struct
import pytest
import nnabla.utils.load as nnload
from nnabla.utils import nnabla_pb2
import onnx
import onnx.helper
from onnx import (ModelProto, TensorProto, GraphProto,
    TensorShapeProto, AttributeProto, NodeProto)
import nnabla.logger as logger
import numpy as np
import pdb
import nnabla.utils.load
import nnabla.utils.network
import onnx_caffe2.backend
from nnabla.utils.converter.nnabla import NnpReader, NnpExporter
#from nnabla.utils.converter.onnx import OnnxReader, OnnxExporter

SKIP_RAW_DATA = False # Skip raw data fro debugging purposes
NNABLA_DOMAIN = "org.nnabla"
MIN_NNABLA_OPSET_VERSION = 1
MIN_ONNX_IR_VERSION = 3
MIN_ONNX_OPSET_VERSION = 2
PRODUCER_NAME = "nnabla-onnx"
PRODUCER_VERSION = "0.1"
TEST_DATA_DIR="conversion_data"

# We default to concat the channel axis
# so the concat results match with caffe2
DEFAULT_CONCAT_AXIS = 1

def onnx_value_info_proto_to_variable(info, network):
    if not info.type.HasField("tensor_type"): # accepting only tensor
        logger.warning("Only TensorProto is allowed as ValueInfoProto's type for now (Got {}). Skipping {}"
                .format(info.type, info.name))
        return
    t = info.type.tensor_type
    v = network.variable.add()
    v.name = info.name
    v.shape.dim.extend([x.dim_value for x in t.shape.dim])
    return v

# Dictionary used to convert ONNX op_type to NNabla function names
onnx_optype_to_nnabla_function_type = {
    "Relu": "ReLU",
    "Concat": "Concatenate",
    "Conv": "Convolution",
    "GlobalAveragePool": "AveragePooling",
    "MaxPool": "MaxPooling",
}

# Dictionary used to convert NNabla function names to ONNX op_type 
nnabla_function_type_to_onnx_optype = {
    "ReLU": "Relu",
    "Concatenate": "Concat",
    "Convolution": "Conv",
    "AveragePooling": "GlobalAveragePool",
    "MaxPooling": "MaxPool",
}

def convert_to_function(node):
    """Convert given node to corresponding function"""
    func = nnabla_pb2.Function()
    func.name = node.name
    func.type = onnx_optype_to_nnabla_function_type.get(node.op_type, node.op_type)
    func.input.extend(node.input)
    func.output.extend(node.output)
    if node.op_type == "Concat":
        # Since concat axis is currently not required in ONNX,
        # the default axis depends on which backend we use.
        # For now we are comparing with caffe2, so we are 
        # defaulting to the channel axis if the axis is not specified.
        # https://github.com/onnx/onnx/issues/374
        axis_count = 0
        for attr in node.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Axis type must be a single integer")
                # The axis was specified so we use it
                func.concatenate_param.axis = attr.i
                axis_count += 1
        if axis_count == 0:
            # No axis was specifed so we default to the channel axis for now
            func.concatenate_param.axis = DEFAULT_CONCAT_AXIS
        elif axis_count > 1:
            raise ValueError("More than one axis was specifed as the Concat Axis")
    elif node.op_type == "Softmax":
        func.type = "Identity"
        # default to channel axis
        #func.softmax_param.axis = 1
        #for attr in node.attribute:
        #    if attr.name == "axis":
        #        if attr.type != AttributeProto.INT:
        #            raise ValueError("Softmax axis must be a single integer")
        #        func.softmax_param.axis = attr.i
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
    elif node.op_type == "GlobalAveragePool":
        func.type = "Identity"
        ## We substitute GlobalAveragePool with an AveragePool
        ## that has the same kernel size as the input WxH
        #app = func.average_pooling_param
        #app.kernel.dim.extend([3,3])
        #app.stride.dim.extend([3,3])
        #app.pad.dim.extend([0,0])
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
        # Pooling should ignore borders when a valid padding value
        # has been set in order to match the ONNX results
        ignore_border = pads and (pads[0] > 0)
        mpp.ignore_border = ignore_border

    return func


def onnx_graph_to_nnp_protobuf(pb, graph):
    network = pb.network.add()
    network.name = graph.name

    all_vars = set()
    # convert nodes
    for n in graph.node:
        f = convert_to_function(n)
        #Gather all unique names for input and output
        for i in f.input:
            all_vars.add(i)
        for o in f.output:
            all_vars.add(o)
        network.function.extend([f])

    # convert parameters
    param_vars = set()
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            logger.warning("Only floating point data is supported for parameters (Got {}). Skipping {}"
                    .format(init.data_type, init.name))
            pass
        p = pb.parameter.add()
        p.variable_name = init.name
        p.shape.dim.extend(init.dims)
        if not SKIP_RAW_DATA:
            # convert raw bytestream to floating points
            num = len(init.raw_data) // 4
            #logger.log(99, "raw_data num: {}".format(num))
            data = struct.unpack(str(num)+'f', init.raw_data)
            p.data.extend(data)
        p.need_grad = False
        # Keep the list of all initializer names
        param_vars.add(init.name)
    # We need to distinguish constant parameters (which become 'Parameter' in NNabla)
    # from input/output variables (which become 'Buffer' in NNabla).
    # Contant parameters appear in the initializer list so we keep
    # all names of variables from the initializer and compare them with 
    # the names we gathered in all_vars.
    # The names that only appear in all_vars are the input/output variables.

    # convert Input/Output ValueInfoProto
    # to Variable
    in_list = []
    param_list = []
    out_list = []
    for i in graph.input:
        v = onnx_value_info_proto_to_variable(i, network)
        if v.name in param_vars:
            # This input is a parameter
            v.type = "Parameter"
            v.initializer.type = "Constant"
            v.initializer.multiplier = 1.0
            param_list.append(v)
        else :
            # This input is a variable
            v.type ="Buffer"
            in_list.append(v)
        all_vars.remove(v.name)
    for o in graph.output:
        v = onnx_value_info_proto_to_variable(o, network)
        v.type = "Buffer"
        out_list.append(v)
        all_vars.remove(v.name)

    for varg in all_vars:
        # We add all remaining variables as intermediate buffer
        v = network.variable.add()
        v.type = "Buffer"
        v.name = varg
        # We calculate the buffer size of all intermediate buffers here
    
    #pdb.set_trace()

    # Add executor for target network
    exe = pb.executor.add()
    exe.name = "exec_0"
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
            logger.warning("Unknown opset from domain {}. Ignoring.".format(opset.domain))

    # convert onnx model to nnabla protobuf
    #logger.log(99, "Converting ONNX made by {}.".format(model.producer_name))

    # conver graph
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

def convert_to_node(func, variables):
    n = onnx.helper.make_node(
            nnabla_function_type_to_onnx_optype.get(func.type, func.type),
            func.input,
            func.output,
            name=func.name)
    if func.type == "Concatenate":
        # ONNX requires axis setting as a parameter
        # for the concat op_type.
        # If no value is set for axis,
        # the default value 0 will be set
        attr = onnx.helper.make_attribute("axis", func.concatenate_param.axis)
        n.attribute.extend([attr])
    elif func.type == "Dropout":
        # NNP Dropout is always is_test=false
        # since we always apply dropout when it is
        # included in a network.
        attr = onnx.helper.make_attribute("is_test", 0)
        n.attribute.extend([attr])
    elif func.type == "Identity":
        # Convert Identity to a Dropout with is_test=true
        # so we just copy the input to output
        n.op_type = "Dropout"
        attr = onnx.helper.make_attribute("is_test", 1)
        n.attribute.extend([attr])
    elif func.type == "MaxPooling":
        mpp = func.max_pooling_param
        # Copy kernel, stride, and pads values
        k = onnx.helper.make_attribute("kernel_shape", mpp.kernel.dim)
        s = onnx.helper.make_attribute("strides", mpp.stride.dim)
        p = onnx.helper.make_attribute("pads", mpp.pad.dim)
        n.attribute.extend([k, s, p])
    elif func.type == "Convolution":
        cp = func.convolution_param
        # Calculate the kernel_shape from input weight data.
        # Weight data should be the second input for convolution
        if len(func.input) < 2:
            raise ValueError(
                "Weight input is missing for convolution {}"
                .format(func.name))
        weight = func.input[1]
        weight_var = [v for v in variables if v.name == weight]
        if len(weight_var) != 1:
            raise ValueError(
                "Multiple weight inputs were found for convolution {} where there should be only one."
                .format(func.name))
        weight_shape = weight_var[0].shape
        # The base axis for weights is the next axis from the data's base axis
        weight_base = cp.base_axis + 1
        k = onnx.helper.make_attribute("kernel_shape", weight_shape.dim[weight_base:])
        d = onnx.helper.make_attribute("dilations", cp.dilation.dim)
        s = onnx.helper.make_attribute("strides", cp.stride.dim)
        p = onnx.helper.make_attribute("pads", cp.pad.dim)
        g = onnx.helper.make_attribute("group", cp.group)
        n.attribute.extend([k, d, s, p, g])
    return n

def nnp_model_to_onnx_graph(graph, nnp):
    if len(nnp.network) != 1:
        raise ValueError("NNP with only a single network is currently supported")
    if len(nnp.executor) != 1:
        raise ValueError("NNP with only a single executor is currently supported")
    net = nnp.network[0]
    exe = nnp.executor[0]
    if exe.network_name != net.name:
        raise ValueError("Names of the included network and executor's target network do not match")
    graph.name = net.name
    # store all variable shape info to use later
    var_dict = {}
    for v in net.variable:
        var_dict[v.name] = v.shape

    for f in net.function:
        n = convert_to_node(f, net.variable)
        graph.node.extend([n])
    for param in nnp.parameter:
        init = graph.initializer.add()
        init.name = param.variable_name
        init.dims.extend(param.shape.dim)
        init.data_type = TensorProto.FLOAT # We should be only getting float data from NNabla
        init.raw_data = struct.pack("{}f".format(len(param.data)), *param.data)
        #init.float_data.extend(param.data)
    # Add all the constant parameters for all nodes
    # and the first node's input as input
    def create_dim(d):
        """Createa dimension message for a given dimension"""
        dim = TensorShapeProto.Dimension()
        dim.dim_value = d
        return dim

    for iv in exe.data_variable:
        i = graph.input.add()
        i.name = iv.variable_name
        i.type.tensor_type.elem_type = TensorProto.FLOAT
        dims = [create_dim(d) for d in var_dict[iv.variable_name].dim]
        i.type.tensor_type.shape.dim.extend(dims)
    for pv in exe.parameter_variable:
        p = graph.input.add()
        p.name = pv.variable_name
        p.type.tensor_type.elem_type = TensorProto.FLOAT
        dims = [create_dim(d) for d in var_dict[pv.variable_name].dim]
        p.type.tensor_type.shape.dim.extend(dims)
    # Add only the final output of the graph as output
    for ov in exe.output_variable:
        o = graph.output.add()
        o.name = ov.variable_name
        o.type.tensor_type.elem_type = TensorProto.FLOAT
        dims = [create_dim(d) for d in var_dict[ov.variable_name].dim]
        o.type.tensor_type.shape.dim.extend(dims)

def nnp_model_to_onnx_protobuf(nnp):
    mp = ModelProto()
    mp.ir_version = MIN_ONNX_IR_VERSION
    opset = mp.opset_import.add()
    opset.version = MIN_ONNX_OPSET_VERSION
    #nn_opset = mp.opset_import.add()
    #nn_opset.domain = NNABLA_DOMAIN
    #nn_opset.version = MIN_NNABLA_OPSET_VERSION
    mp.producer_name = PRODUCER_NAME
    mp.producer_version = PRODUCER_VERSION
    mp.domain = NNABLA_DOMAIN
    nnp_model_to_onnx_graph(mp.graph, nnp)
    return mp

class OnnxExporter:
    def __init__(self, nnp):
        self._nnp = nnp.protobuf

    def export(self, file_path):
        model_proto = nnp_model_to_onnx_protobuf(self._nnp)
        with open(file_path, "wb") as f:
            f.write(model_proto.SerializeToString())


def run_executor(nn_net, exec_name):
    """Run specified executor and return its network"""
    exe = nn_net.executors[exec_name]
    exe.network.forward(exe.forward_sequence)
    return exe.network


def convert_onnx_to_nnp_and_compare(
        tmpdir, onnx_dir, onnx_name, nnp_name, out_name, exec_name,
        compare_values=True, show_onnx=False, show_nnp=False, show_output=False):
    """Convert specified ONNX to NNP and compare each results ran by Caffe2 and NNabla"""
    path = os.path.join(onnx_dir, onnx_name)
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    if show_onnx:
        print(model)
    c2out = onnx_caffe2.backend.run_model(model, [])
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    if show_nnp:
        print(nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), nnp_name)
    nnpex.export_nnp(p)
    # read exported nnp and run network
    #pdb.set_trace()
    nn_net = nnload.load([p])
    exe = run_executor(nn_net, exec_name)
    #in_data = exe.variables["in_data_0"]
    #print(in_data.variable_instance.d)
    nnout = exe.variables[out_name].variable_instance.d
    #print(nnout.variable_instance.d)
    # Compare both naabla and caffe2 results
    c2 = c2out[out_name]
    if show_output:
        print(c2, nnout)
    assert c2.shape == nnout.shape
    if compare_values:
        assert np.allclose(c2, nnout)

def convert_nnp_to_onnx_and_compare(
        tmpdir, nnp_dir, nnp_name, onnx_name, out_name, exec_name,
        compare_values=True, show_nnp=False, show_onnx=False, show_output=False):
    """Convert specified NNP to ONNX and compare each results ran by Caffe2 and NNabla"""
    # Process nnp with nnabla
    path = os.path.join(nnp_dir, nnp_name)
    nn_net = nnload.load([path])
    exe = run_executor(nn_net, exec_name)
    nnout = exe.variables[out_name].variable_instance.d

    # Convert nnp to ONNX
    r = NnpReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    if show_nnp:
        print(nnp.protobuf)
    onnxex = OnnxExporter(nnp)
    onnxdir = tmpdir.mkdir("onnx")
    p = os.path.join(str(onnxdir), onnx_name)
    onnxex.export(p)

    # read exported onnx and run network
    model = onnx.load(p)
    if show_onnx:
        print(model)
    #pdb.set_trace()
    c2out = onnx_caffe2.backend.run_model(model, [])
    c2 = c2out[out_name]
    # Compare both naabla and caffe2 results
    if show_output:
        print(c2, nnout)
    assert c2.shape == nnout.shape
    if compare_values:
        assert np.allclose(c2, nnout)

@pytest.fixture
def nnp_fixture():
    # We need to remove all parameters for each test case
    # because the buffer shape will differ while having same names
    nnabla.clear_parameters()

def test_onnx_nnp_conversion_relu(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "relu.onnx", "relu.nnp", "out_data_1", "exec_0")

def test_nnp_onnx_conversion_relu(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "relu.nnp", "relu.onnx", "out_data_1", "exec_0")

def test_onnx_nnp_conversion_concat(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "concat.onnx", "concat.nnp", "out_data_1", "exec_0")

def test_nnp_onnx_conversion_concat(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "concat.nnp", "concat.onnx", "out_data_1", "exec_0")

def test_onnx_nnp_conversion_dropout(tmpdir, nnp_fixture):
    # We do not check if the values match because a dropout
    # output yield random results
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout.onnx", "dropout.nnp", "out_data_1", "exec_0", compare_values=False)

def test_nnp_onnx_conversion_dropout(tmpdir, nnp_fixture):
    # We do not check if the values match because a dropout
    # output yield random results
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout.nnp", "dropout.onnx", "out_data_1", "exec_0", compare_values=False)

def test_onnx_nnp_conversion_dropout_is_test(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout_test.onnx", "dropout_test.nnp", "out_data_1", "exec_0")

def test_nnp_onnx_conversion_dropout_is_test(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "dropout_test.nnp", "dropout_test.onnx", "out_data_1", "exec_0")

def test_onnx_nnp_conversion_maxpool(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool.onnx", "maxpool.nnp", "out_data_1", "exec_0")

def test_nnp_onnx_conversion_maxpool(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool.nnp", "maxpool.onnx", "out_data_1", "exec_0")

def test_onnx_nnp_conversion_maxpool_no_pad(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_no_pad.onnx", "maxpool_no_pad.nnp", "out_data_1", "exec_0")

def test_nnp_onnx_conversion_maxpool_no_pad(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "maxpool_no_pad.nnp", "maxpool_no_pad.onnx", "out_data_1", "exec_0")

def test_onnx_nnp_conversion_conv(tmpdir, nnp_fixture):
    convert_onnx_to_nnp_and_compare(
        tmpdir, TEST_DATA_DIR, "conv.onnx", "conv.nnp", "out_data_1", "exec_0")

def test_nnp_onnx_conversion_conv(tmpdir, nnp_fixture):
    convert_nnp_to_onnx_and_compare(
        tmpdir, TEST_DATA_DIR, "conv.nnp", "conv.onnx", "out_data_1", "exec_0")

def test_onnx_nnp_conversion_squeezenet(tmpdir, nnp_fixture):
    onnx_dir = TEST_DATA_DIR
    onnx_name = "squeezenet.onnx"
    nnp_name = "squeezenet.nnp"
    out_name = "softmaxout_1"
    exec_name = "exec_0"
    show_onnx = False
    show_nnp = False
    path = os.path.join(onnx_dir, onnx_name)
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    if show_onnx:
        print(model)
    img = np.random.rand(1,3,224,224).astype(np.float32)

    # Remove Softmax and GlobalAveragePooling for now.
    # This is temporal
    nodes = len(model.graph.node)
    sm_node = model.graph.node[nodes-1]
    gap_node = model.graph.node[nodes-2]
    def change_to_copy(node):
        """Change node operation to a simple copy"""
        # Dropout with is_test=True is equal to a simple copy
        node.op_type = "Dropout"
        attr = node.attribute.add()
        attr.name = "is_test"
        attr.type = AttributeProto.INT
        attr.i = 1
    change_to_copy(sm_node)
    change_to_copy(gap_node)
    # Change the output dimension so it matches the actual size
    out_shape = model.graph.output[0].type.tensor_type.shape
    out_shape.dim[0].dim_value = 1
    out_shape.dim[1].dim_value = 1000
    out_shape.dim[2].dim_value = 13
    out_shape.dim[3].dim_value = 13

    c2out = onnx_caffe2.backend.run_model(model, [img])
    # Process onnx with naabla
    nnp = onnx_model_to_nnp_protobuf(model)
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    if show_nnp:
        print(nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), nnp_name)
    nnpex.export_nnp(p)
    #pdb.set_trace()
    # read exported nnp and run network
    #nn_net = nnload.load([p])
    #exe = run_executor(nn_net, exec_name)
    ##in_data = exe.variables["in_data_0"]
    ##print(in_data.variable_instance.d)
    #nnout = exe.variables[out_name].variable_instance.d
    ##print(nnout.variable_instance.d)
    ## Compare both naabla and caffe2 results
    #c2 = c2out[out_name]
    #if show_output:
    #    print(c2, nnout)
    #assert c2.shape == nnout.shape
    #if compare_values:
    #    assert np.allclose(c2, nnout)

#def test_onnx_nnp_conversion_softmax(tmpdir):
#    path = os.path.join(TEST_DATA_DIR, "softmax.onnx")
#    # Process onnx with caffe2 backend
#    model = onnx.load(path)
#    c2out = onnx_caffe2.backend.run_model(model, [])
#    # Process onnx with naabla
#    r = OnnxReader(path)
#    nnp = r.read()
#    assert nnp is not None
#    assert len(nnp.other_files) == 0
#    assert nnp.protobuf is not None
#    logger.log(99, nnp.protobuf)
#
#    nnpex = NnpExporter(nnp, batch_size=0)
#    nnpdir = tmpdir.mkdir("nnp")
#    p = os.path.join(str(nnpdir), "softmax.nnp")
#    nnpex.export_nnp(p)
#    # read exported nnp and run network
#    #pdb.set_trace()
#    nn_net = nnload.load([p])
#    softmax = run_executor(nn_net, "exec_0")
#    OUT_DATA_NAME = "out_data_1"
#    nnout = softmax.variables[OUT_DATA_NAME].variable_instance.d
#    c2 = c2out[OUT_DATA_NAME]
#    #print(softmax.variables["in_data_0"].variable_instance.d)
#    print(np.sum(c2))
#    print(np.sum(nnout))
#    print(c2, c2.shape)
#    print(nnout, nnout.shape)
#    #assert np.allclose(c2, nnout)

##
#def test_onnx_nnp_conversion_gap(tmpdir):
#    path = os.path.join(TEST_DATA_DIR, "gap.onnx")
#    # Process onnx with caffe2 backend
#    model = onnx.load(path)
#    c2out = onnx_caffe2.backend.run_model(model, [])
#    #print(c2out)
#    # Process onnx with naabla
#    r = OnnxReader(path)
#    nnp = r.read()
#    assert nnp is not None
#    assert len(nnp.other_files) == 0
#    assert nnp.protobuf is not None
#    #logger.log(99, nnp.protobuf)
#
#    nnpex = NnpExporter(nnp, batch_size=0)
#    nnpdir = tmpdir.mkdir("nnp")
#    p = os.path.join(str(nnpdir), "gap.nnp")
#    nnpex.export_nnp(p)
#    # read exported nnp and run network
#    #pdb.set_trace()
#    nn_net = nnload.load([p])
#    gap = run_executor(nn_net, "exec_0")
#    OUT_DATA_NAME = "out_data_1"
#    out_data = gap.variables[OUT_DATA_NAME]
#    nnout = gap.variables[OUT_DATA_NAME].variable_instance.d
#    c2 = c2out[OUT_DATA_NAME]
#    #print(c2, nnout)
#    assert np.allclose(c2, nnout)
