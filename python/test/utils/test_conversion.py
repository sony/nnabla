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
import nnabla.utils.converter
import nnabla.utils.load as nnload
from nnabla.utils import nnabla_pb2
import onnx
from onnx import (defs, checker, helper, numpy_helper, mapping,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorSetIdProto)
import onnx_caffe2.backend
import nnabla.logger as logger
import numpy as np
import nnabla as nn
import pdb
from nnabla.utils.converter.nnabla import NnpExporter
#from nnabla.utils.converter.onnx import OnnxReader, OnnxExporter

TEST_DATA_DIR="conversion_data"

def onnx_data_type_to_string(data_type):
    if data_type == TensorProto.FLOAT:
        return "float"
    else:
        return "unknown"

def onnx_value_info_proto_to_variable(info, network, out_list):
    if not info.type.HasField("tensor_type"): # accepting only tensor
        logger.warning("Only TensorProto is allowed as ValueInfoProto's type for now (Got {}). Skipping {}"
                .format(info.type, info.name))
        return
    t = info.type.tensor_type
    v = network.variable.add()
    v.name = info.name
    v.type = onnx_data_type_to_string(t.elem_type)
    v.shape.dim.extend([x.dim_value for x in t.shape.dim])
    out_list.append(v)

def onnx_graph_to_protobuf(pb, graph):
    network = pb.network.add()
    network.name = graph.name
    # convert nodes
    for n in graph.node:
        f = network.function.add()
        f.name = n.name
        f.type = n.op_type
        f.input.extend(n.input)
        f.output.extend(n.output)

    # convert Input/Output ValueInfoProto
    # to Variable
    in_vars = []
    out_vars = []
    mid_vars = []
    for i in graph.input:
        onnx_value_info_proto_to_variable(i, network, in_vars)
    for o in graph.output:
        onnx_value_info_proto_to_variable(o, network, out_vars)
    for vi in graph.value_info:
        onnx_value_info_proto_to_variable(vi, network, mid_vars)

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
        logger.log(99, "raw_data num: {}".format(num))
        data = struct.unpack(str(num)+'f', init.raw_data)
        p.data.extend(data)
        p.need_grad = False

    # Add executor for target network
    exe = pb.executor.add()
    exe.name = "exec_0"
    exe.network_name = graph.name
    for iv in in_vars:
        dv = exe.data_variable.add()
        dv.variable_name = iv.name
        dv.data_name = iv.name
    for ov in out_vars:
        outv = exe.output_variable.add()
        outv.variable_name = ov.name
        outv.data_name = ov.name


def onnx_model_to_protobuf(model):
    pb = nnabla_pb2.NNablaProtoBuf()
    if model.ir_version > 3:
        raise ValueError("ONNX models newer than version 3 is currently not supported")
    for opset in model.opset_import:
        if opset.domain == "":
            # ONNX opset.
            # Check if we have the correct version
            if opset.version > 2:
                raise ValueError("ONNX opsets newer than version 2 is currently not supported")
        else:
            logger.warning("Unknown opset from domain {}. Ignoring.".format(opset.domain))

    # convert onnx model to nnabla protobuf
    logger.log(99, "Converting ONNX made by {}.".format(model.producer_name))

    # conver graph
    onnx_graph_to_protobuf(pb, model.graph)

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
        return onnx_model_to_protobuf(model_proto)

def test_onnx_nnp_conversion_relu(tmpdir):
    path = os.path.join(TEST_DATA_DIR, "relu.onnx")
    # Process onnx with caffe2 backend
    #model = onnx.load(path)
    #c2out = onnx_caffe2.backend.run_model(model, [])
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    assert len(nnp.other_files) == 0
    assert nnp.protobuf is not None
    logger.log(99, nnp.protobuf)

    nnpex = NnpExporter(nnp, batch_size=0)
    nnpdir = tmpdir.mkdir("nnp")
    p = os.path.join(str(nnpdir), "relu.nnp")
    #pdb.set_trace()
    nnpex.export_nnp(p)
    # read exported nnp and run network
    nn_net = nnload.load(p)
    # Compare both naabla and caffe2 results
    #assert np.allclose(c2out, nout)
