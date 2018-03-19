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
import nnabla.logger as logger
import onnx
from .utils import *
from onnx import (ModelProto, TensorProto, TensorShapeProto)

# Dictionary used to convert NNabla function names to ONNX op_type
nnabla_function_type_to_onnx_optype = {
    "ReLU": "Relu",
    "Concatenate": "Concat",
    "Convolution": "Conv",
    "GlobalAveragePooling": "GlobalAveragePool",
    "MaxPooling": "MaxPool",
}


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
        if not mpp.ignore_border:
            raise ValueError(
                "MaxPooling with ignore_border=False is not supported")
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
                "No weight input was found, or multiple weight inputs were found"
                " for convolution {} where there should be only one."
                .format(func.name))
        weight_shape = weight_var[0].shape
        # The base axis for weights is the next axis from the data's base axis
        weight_base = cp.base_axis + 1
        k = onnx.helper.make_attribute(
            "kernel_shape", weight_shape.dim[weight_base:])
        d = onnx.helper.make_attribute("dilations", cp.dilation.dim)
        s = onnx.helper.make_attribute("strides", cp.stride.dim)
        p = onnx.helper.make_attribute("pads", cp.pad.dim)
        g = onnx.helper.make_attribute("group", cp.group)
        n.attribute.extend([k, d, s, p, g])
    elif func.type == "GlobalAveragePooling":
        # We wipeout the node name to avoid a bug?
        # that occurs when we use a GlobalAveragePooling node with a name
        # "Conv" or "Pool" contained.
        # Caffe2 issue is here:
        # https://github.com/caffe2/caffe2/issues/1971
        # Becuase a GlobalAveragePooling operator does not contain a kernel, we get an error at the
        # following code if we have a specific name.
        # https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_pool_op_base.h#L167
        # The above caffe2 code should be checking the node's operator name and not the node's name.
        n.name = ""
    elif func.type == "Softmax":
        # Softmax on NNabla does softmax ONLY along the specified axis.
        # ONNX first squashes the input dimensions to 2D based on the specifed axis,
        # and then calculates the Softmax.
        # Since these two slightly differ, we show a warning here.
        logger.warning(SOFTMAX_WARNING)
        attr = onnx.helper.make_attribute("axis", func.softmax_param.axis)
        n.attribute.extend([attr])
    return n


def nnp_model_to_onnx_graph(graph, nnp):
    if len(nnp.network) != 1:
        raise ValueError(
            "NNP with only a single network is currently supported")
    if len(nnp.executor) != 1:
        raise ValueError(
            "NNP with only a single executor is currently supported")
    net = nnp.network[0]
    exe = nnp.executor[0]
    if exe.network_name != net.name:
        raise ValueError(
            "Names of the included network and executor's target network do not match")
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
        # We should be only getting float data from NNabla
        init.data_type = TensorProto.FLOAT
        init.raw_data = struct.pack("{}f".format(len(param.data)), *param.data)
        # init.float_data.extend(param.data)

    # Add all the constant parameters for all nodes
    # and the first node's input as input
    def create_dim(val):
        """Create a dimension message for a given dimension"""
        dim = TensorShapeProto.Dimension()
        dim.dim_value = val
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
    op0 = mp.opset_import.add()
    op0.version = MIN_ONNX_OPSET_VERSION
    op1 = mp.opset_import.add()
    op1.domain = ""  # empty string indicates ONNX domain
    op1.version = MIN_ONNX_OPSET_VERSION
    # nn_opset = mp.opset_import.add()
    # nn_opset.domain = NNABLA_DOMAIN
    # nn_opset.version = MIN_NNABLA_OPSET_VERSION
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
