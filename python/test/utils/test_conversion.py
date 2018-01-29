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
import nnabla.utils.load as nnload
from nnabla.utils import nnabla_pb2
import onnx
from onnx import (ModelProto, TensorProto, GraphProto, TensorShapeProto, AttributeProto)
import nnabla.logger as logger
import numpy as np
import pdb
import nnabla.utils.load
import nnabla.utils.network
import onnx_caffe2.backend
from nnabla.utils.converter.nnabla import NnpReader, NnpExporter
#from nnabla.utils.converter.onnx import OnnxReader, OnnxExporter

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
    "GlobalAveragePool": "AveragePooling"
}

# Dictionary used to convert NNabla function names to ONNX op_type 
nnabla_function_type_to_onnx_optype = {
    "ReLU": "Relu",
    "Concatenate": "Concat",
    "Convolution": "Conv", 
    "AveragePooling": "GlobalAveragePool",
}

def convert_to_function(node):
    '''Convert given node to corresponding function'''
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
        # default to channel axis
        func.softmax_param.axis = 1
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
                    func.type = "Identity"
                    # We break here so we don't write any needless attributes
                    break
            elif attr.name == "ratio":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Dropout ratio must be a single float")
                func.dropout_param.p = attr.f
    elif node.op_type == "Conv":
        func.convolution_param.base_axis = 2
        for attr in node.attribute:
            #if attr.name == "kernel_shape":
            #    if attr.type != AttributeProto.INTS:
            #        raise ValueError("Only INTS are supported for kernel_shape in Conv op_type")
            #    func.convolution_param.dilation.dim.extend(attr.ints)
            if attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for pads in Conv op_type")
                func.convolution_param.pad.dim.extend(attr.ints)
            elif attr.name == "strides":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for strides in Conv op_type")
                func.convolution_param.stride.dim.extend(attr.ints)
    elif node.op_type == "GlobalAveragePool":
        # We substitute GlobalAveragePool with an AveragePool
        # that has the same kernel size as the input WxH
        app = func.average_pooling_param
        app.kernel.dim.extend([3,3])
        app.stride.dim.extend([3,3])
        app.pad.dim.extend([0,0])
    return func


def onnx_graph_to_nnp_protobuf(pb, graph):
    network = pb.network.add()
    network.name = graph.name

    # convert nodes
    for n in graph.node:
        f = convert_to_function(n)
        network.function.extend([f])

    # convert Input/Output ValueInfoProto
    # to Variable
    in_vars = []
    param_vars = []
    out_vars = []
    mid_vars = []
    for i in graph.input:
        v = onnx_value_info_proto_to_variable(i, network)
        v.type = "Parameter"
        v.initializer.type = "Constant"
        v.initializer.multiplier = 1.0
        param_vars.append(v)
    for o in graph.output:
        v = onnx_value_info_proto_to_variable(o, network)
        v.type = "Buffer"
        out_vars.append(v)
    for vi in graph.value_info:
        v = onnx_value_info_proto_to_variable(vi, network)
        mid_vars.append(v)

    # convert parameters
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            logger.warning("Only floating point data is supported for parameters (Got {}). Skipping {}"
                    .format(init.data_type, init.name))
            pass
        p = pb.parameter.add()
        p.variable_name = init.name
        p.shape.dim.extend(init.dims)
        # convert raw bytestream to floating points
        num = len(init.raw_data) // 4
        #logger.log(99, "raw_data num: {}".format(num))
        data = struct.unpack(str(num)+'f', init.raw_data)
        p.data.extend(data)
        p.need_grad = False

    # Add executor for target network
    exe = pb.executor.add()
    exe.name = "exec_0"
    exe.network_name = network.name
    for iv in in_vars:
        dv = exe.data_variable.add()
        dv.variable_name = iv.name
        dv.data_name = iv.name
    for ov in out_vars:
        outv = exe.output_variable.add()
        outv.variable_name = ov.name
        outv.data_name = ov.name
    for pv in param_vars:
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

def set_node_attribute(node, func):
    if func.type == "Concatenate":
        # ONNX requires axis setting as a parameter
        # for the concat op_type.
        attr = node.attribute.add()
        attr.name = "axis"
        attr.type = AttributeProto.INT
        # If no value is set for axis,
        # the default value 0 will be set
        attr.i = func.concatenate_param.axis

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
        n = graph.node.add()
        n.name = f.name
        n.op_type = nnabla_function_type_to_onnx_optype.get(f.type, f.type)
        n.input.extend(f.input)
        n.output.extend(f.output)
        set_node_attribute(n, f)
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
        '''Createa dimension message for a given dimension'''
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
    '''Run specified executor and return its network'''
    exe = nn_net.executors[exec_name]
    exe.network.forward(exe.forward_sequence)
    return exe.network

def test_onnx_nnp_conversion_relu(tmpdir):
    path = os.path.join(TEST_DATA_DIR, "relu.onnx")
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    c2out = onnx_caffe2.backend.run_model(model, [])
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    #logger.log(99, nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), "relu.nnp")
    nnpex.export_nnp(p)
    # read exported nnp and run network
    #pdb.set_trace()
    nn_net = nnload.load([p])
    relu = run_executor(nn_net, "exec_0")
    #in_data = relu.variables["in_data_0"]
    #print(in_data.variable_instance.d)
    OUT_DATA_NAME = "out_data_1"
    nnout = relu.variables[OUT_DATA_NAME].variable_instance.d
    #print(nnout.variable_instance.d)
    # Compare both naabla and caffe2 results
    c2 = c2out[OUT_DATA_NAME]
    #print(c2, nnout)
    assert np.allclose(c2, nnout)

def test_nnp_onnx_conversion_relu(tmpdir):
    # Process nnp with nnabla
    OUT_DATA_NAME = "out_data_1"
    path = os.path.join(TEST_DATA_DIR, "relu.nnp")
    nn_net = nnload.load([path])
    relu = run_executor(nn_net, "exec_0")
    nnout = relu.variables[OUT_DATA_NAME].variable_instance.d

    # Convert nnp to ONNX
    r = NnpReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    #logger.log(99, nnp.protobuf)
    onnxex = OnnxExporter(nnp)
    onnxdir = tmpdir.mkdir("onnx")
    p = os.path.join(str(onnxdir), "relu.onnx")
    onnxex.export(p)

    # read exported onnx and run network
    model = onnx.load(p)
    #print(model)
    #pdb.set_trace()
    c2out = onnx_caffe2.backend.run_model(model, [])
    c2 = c2out[OUT_DATA_NAME]
    # Compare both naabla and caffe2 results
    #print(c2, nnout)
    assert np.allclose(c2, nnout)

def test_onnx_nnp_conversion_concat(tmpdir):
    path = os.path.join(TEST_DATA_DIR, "concat.onnx")
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    c2out = onnx_caffe2.backend.run_model(model, [])
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    #logger.log(99, nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), "concat.nnp")
    nnpex.export_nnp(p)
    # read exported nnp and run network
    #pdb.set_trace()
    nn_net = nnload.load([p])
    concat = run_executor(nn_net, "exec_0")
    #id0 = concat.variables["in_data_0_0"]
    #id1 = concat.variables["in_data_1_0"]
    #print(id0.variable_instance.d)
    #print(id1.variable_instance.d)
    OUT_DATA_NAME = "out_data_1"
    nnout = concat.variables[OUT_DATA_NAME].variable_instance.d
    c2 = c2out[OUT_DATA_NAME]
    #print(c2, c2.shape)
    #print(nnout, nnout.shape)
    assert np.allclose(c2, nnout)

def test_nnp_onnx_conversion_concat(tmpdir):
    # Process nnp with nnabla
    OUT_DATA_NAME = "out_data_1"
    path = os.path.join(TEST_DATA_DIR, "concat.nnp")
    nn_net = nnload.load([path])
    relu = run_executor(nn_net, "exec_0")
    nnout = relu.variables[OUT_DATA_NAME].variable_instance.d

    # Convert nnp to ONNX
    r = NnpReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    #logger.log(99, nnp.protobuf)
    onnxex = OnnxExporter(nnp)
    onnxdir = tmpdir.mkdir("onnx")
    p = os.path.join(str(onnxdir), "concat.onnx")
    onnxex.export(p)

    # read exported onnx and run network
    model = onnx.load(p)
    #print(model)
    #pdb.set_trace()
    c2out = onnx_caffe2.backend.run_model(model, [])
    c2 = c2out[OUT_DATA_NAME]
    # Compare both naabla and caffe2 results
    #print(c2.shape, nnout.shape)
    assert np.allclose(c2, nnout)

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

def test_onnx_nnp_conversion_dropout(tmpdir):
    path = os.path.join(TEST_DATA_DIR, "dropout.onnx")
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    c2out = onnx_caffe2.backend.run_model(model, [])
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    #logger.log(99, nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), "dropout.nnp")
    nnpex.export_nnp(p)
    # read exported nnp and run network
    #pdb.set_trace()
    nn_net = nnload.load([p])
    dropout = run_executor(nn_net, "exec_0")
    OUT_DATA_NAME = "out_data_1"
    nnout = dropout.variables[OUT_DATA_NAME].variable_instance.d
    c2 = c2out[OUT_DATA_NAME]
    #print(c2, c2.shape)
    #print(nnout, nnout.shape)
    assert c2.shape == nnout.shape
    # We do not check if the values match because a dropout
    # output yield random results
    #assert np.allclose(c2, nnout)

def test_onnx_nnp_conversion_dropout_is_test(tmpdir):
    path = os.path.join(TEST_DATA_DIR, "dropout_test.onnx")
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    c2out = onnx_caffe2.backend.run_model(model, [])
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    #logger.log(99, nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), "dropout_test.nnp")
    nnpex.export_nnp(p)
    # read exported nnp and run network
    #pdb.set_trace()
    nn_net = nnload.load([p])
    dropout = run_executor(nn_net, "exec_0")
    OUT_DATA_NAME = "out_data_1"
    nnout = dropout.variables[OUT_DATA_NAME].variable_instance.d
    c2 = c2out[OUT_DATA_NAME]
    #print(c2, c2.shape)
    #print(nnout, nnout.shape)
    assert np.allclose(c2, nnout)
#
#def test_onnx_nnp_conversion_conv(tmpdir):
#    path = os.path.join(TEST_DATA_DIR, "conv.onnx")
#    # Process onnx with caffe2 backend
#    #model = onnx.load(path)
#    #c2out = onnx_caffe2.backend.run_model(model, [])
#    # Process onnx with naabla
#    r = OnnxReader(path)
#    nnp = r.read()
#    assert nnp is not None
#    assert len(nnp.other_files) == 0
#    assert nnp.protobuf is not None
#    logger.log(99, nnp.protobuf)
#
#    #nnpex = NnpExporter(nnp, batch_size=0)
#    #nnpdir = tmpdir.mkdir("nnp")
#    #p = os.path.join(str(nnpdir), "conv.nnp")
#    #nnpex.export_nnp(p)
#    # read exported nnp and run network
#    # pdb.set_trace()
#    #nn_net = nnload.load([p])
#    #conv = run_executor(nn_net, "exec_0")
#    #OUT_DATA_NAME = "out_data_1"
#    #nnout = conv.variables[OUT_DATA_NAME].variable_instance.d
#    #c2 = c2out[OUT_DATA_NAME]
#    ##print(c2, c2.shape)
#    ##print(nnout, nnout.shape)
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
