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

import nnabla.logger as logger
import onnx
import numpy as np
from .utils import *
from onnx import (ModelProto, TensorProto, TensorShapeProto)

# Dictionary used to convert NNabla function names to ONNX op_type 
nnabla_function_type_to_onnx_optype = {
    # optype with same names
    "Dropout": "Dropout",
    "Softmax": "Softmax",
    "BatchNormalization": "BatchNormalization",
    "Transpose": "Transpose",
    "Abs": "Abs",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "LeakyReLU": "LeakyRelu",
    "Log": "Log",
    # optype with different names
    "ReLU": "Relu",
    "Concatenate": "Concat",
    "Convolution": "Conv",
    "GlobalAveragePooling": "GlobalAveragePool",
    "MaxPooling": "MaxPool",
    "AveragePooling": "AveragePool",
    "Add2": "Add",
    "BatchMatmul": "MatMul",
    "LogicalNot": "Not",
    "ELU": "Elu",
    "SELU": "Selu",
    "Sum": "ReduceSum",
    "Mean": "ReduceMean",
    # optype that gets converted
    "Identity": "Dropout",
    "Affine": "Gemm",
    "Mul2": "Mul",
    # optype that should get merged
    # with other operators
    "BroadcastTo": ""
}

def merge_broadcast(node, func, target_name, broadcast_target):
    # Set the broadcast attribute to the operator
    # so we can combine BroadcastTo with this operator.
    param = broadcast_target[target_name]
    before_broadcast = param[0]
    axis = param[1]
    a = onnx.helper.make_attribute("axis", axis)
    b = onnx.helper.make_attribute("broadcast", 1)
    node.attribute.extend([a, b])
    # Replace the broadcasted input with the original input
    del node.input[:]
    node.input.extend([func.input[0], before_broadcast])
    # Remove the used target.
    # We may have a problem if the same parameter is used from
    # multipler operators.
    del broadcast_target[target_name]

def convert_to_nodes(func, variables, input_types, output_types, broadcast_target):
    """Convert a function to a node or a group of nodes"""
    op_type = nnabla_function_type_to_onnx_optype.get(func.type)
    if op_type is None:
        raise ValueError("function {} is currently not supported for ONNX conversion".format(func.type))
    n = onnx.helper.make_node(
            op_type,
            func.input,
            func.output,
            name=func.name)
    nl = []
    if func.type == "Concatenate":
        # ONNX requires axis setting as a parameter
        # for the concat op_type.
        # If no value is set for axis,
        # the default value 0 will be set
        attr = onnx.helper.make_attribute("axis", func.concatenate_param.axis)
        n.attribute.extend([attr])
        nl.append(n)
    elif func.type == "Dropout":
        # NNP Dropout is always is_test=false
        # since we always apply dropout when it is
        # included in a network.
        attr = onnx.helper.make_attribute("is_test", 0)
        n.attribute.extend([attr])
        nl.append(n)
    elif func.type == "Identity":
        # Convert Identity to a Dropout with is_test=true
        # so we just copy the input to output
        attr = onnx.helper.make_attribute("is_test", 1)
        n.attribute.extend([attr])
        nl.append(n)
    elif func.type == "MaxPooling":
        mpp = func.max_pooling_param
        if not mpp.ignore_border:
            raise ValueError("MaxPooling with ignore_border=False is not supported")
        # Copy kernel, stride, and pads values
        k = onnx.helper.make_attribute("kernel_shape", mpp.kernel.dim)
        s = onnx.helper.make_attribute("strides", mpp.stride.dim)
        p = onnx.helper.make_attribute("pads", mpp.pad.dim*2)
        n.attribute.extend([k, s, p])
        nl.append(n)
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
        k = onnx.helper.make_attribute("kernel_shape", weight_shape.dim[weight_base:])
        d = onnx.helper.make_attribute("dilations", cp.dilation.dim)
        s = onnx.helper.make_attribute("strides", cp.stride.dim)
        p = onnx.helper.make_attribute("pads", cp.pad.dim*2)
        g = onnx.helper.make_attribute("group", cp.group)
        n.attribute.extend([k, d, s, p, g])
        nl.append(n)
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
        nl.append(n)
    elif func.type == "Softmax":
        # Softmax on NNabla does softmax ONLY along the specified axis.
        # ONNX first squashes the input dimensions to 2D based on the specifed axis,
        # and then calculates the Softmax.
        # Since these two slightly differ, we show a warning here.
        logger.warning(SOFTMAX_WARNING)
        attr = onnx.helper.make_attribute("axis", func.softmax_param.axis)
        n.attribute.extend([attr])
        nl.append(n)
    elif func.type == "AveragePooling":
        app = func.average_pooling_param
        if not app.ignore_border:
            raise ValueError("AveragePooling with ignore_border=False is not supported")
        # Copy kernel, stride, and pads values
        k = onnx.helper.make_attribute("kernel_shape", app.kernel.dim)
        s = onnx.helper.make_attribute("strides", app.stride.dim)
        p = onnx.helper.make_attribute("pads", app.pad.dim*2)
        n.attribute.extend([k, s, p])
        nl.append(n)
    elif func.type == "BatchNormalization":
        # We need to rearrange the input data order.
        # NNabla BatchNormalization input order: X, beta, gamma, mean, variance
        # ONNX BatchNormalization input order: X, scale, bias, mean, variance
        onnx_order = [0, 2, 1, 3, 4]
        if len(func.input) != len(onnx_order):
            raise ValueError("The number of BatchNormalization input must be {}".format(len(onnx_order)))
        onnx_input = [func.input[i] for i in onnx_order]
        del n.input[:]
        n.input.extend(onnx_input)
        bpp = func.batch_normalization_param
        if bpp.batch_stat:
            # Batch normalization for training is currently not supported
            raise ValueError("BatchNormalization with batch_stat=True is currently not supported for ONNX conversion")
        t = onnx.helper.make_attribute("is_test", not bpp.batch_stat)
        attrs = [t]
        # Set values if a valid value has been set
        if bpp.eps != 0.0:
            e = onnx.helper.make_attribute("epsilon", bpp.eps)
            attrs.append(e)
        if bpp.decay_rate != 0.0:
            m = onnx.helper.make_attribute("momentum", bpp.decay_rate)
            attrs.append(m)
        n.attribute.extend(attrs)
        nl.append(n)
    elif func.type == "Transpose":
        tp = func.transpose_param
        p = onnx.helper.make_attribute("perm", tp.axes)
        n.attribute.extend([p])
        nl.append(n)
    elif func.type == "Affine":
        ap = func.affine_param
        flatten_postfix = "_flatten"
        # Broadcast tensor C by default since it's usually a 1D vector
        b = onnx.helper.make_attribute("broadcast", 1)
        n.attribute.extend([b])
        # We need to flatten tensor A to 2D based on the base_axis
        x = func.input[0]
        flout = x+flatten_postfix
        fl = onnx.helper.make_node(
                "Flatten",
                [x],
                [flout])
        n.input[0] = flout  # rewire input data
        a = onnx.helper.make_attribute("axis", ap.base_axis)
        fl.attribute.extend([a])
        nl.append(fl)
        nl.append(n)
    elif func.type == "BatchMatmul":
        bmp = func.batch_matmul_param
        if bmp.transpose_a or bmp.transpose_b:
            raise ValueError("{} with transpose is not supported yet".format(func.type))
        nl.append(n)
    elif func.type == "LeakyReLU":
        lrp = func.leaky_relu_param
        a = onnx.helper.make_attribute("alpha", lrp.alpha)
        n.attribute.extend([a])
        nl.append(n)
    elif func.type == "ELU":
        ep = func.elu_param
        a = onnx.helper.make_attribute("alpha", ep.alpha)
        n.attribute.extend([a])
        nl.append(n)
    elif func.type == "LogicalNot":
        # Store the input/output tensor's name and convert it to boolean
        input_types[n.input[0]] = TensorProto.BOOL
        output_types[n.output[0]] = TensorProto.BOOL
        nl.append(n)
    elif func.type == "SELU":
        sp = func.selu_param
        a = onnx.helper.make_attribute("alpha", sp.alpha)
        g = onnx.helper.make_attribute("gamma", sp.scale)
        n.attribute.extend([a, g])
        nl.append(n)
    elif func.type == "Sum":
        sp = func.sum_param
        a = onnx.helper.make_attribute("axes", sp.axes)
        k = onnx.helper.make_attribute("keepdims", sp.keep_dims)
        n.attribute.extend([a, k])
        nl.append(n)
    elif func.type == "Mean":
        mp = func.mean_param
        a = onnx.helper.make_attribute("axes", mp.axes)
        k = onnx.helper.make_attribute("keepdims", mp.keep_dims)
        n.attribute.extend([a, k])
        nl.append(n)
    elif func.type == "BroadcastTo":
        # BroadcastTo conversion only works when the
        # broadcasted buffer is used as second input for the following:
        # Add, And, Div, Equal, Greater,
        # Less, Mul, Or, Pow, Sub, Xor
        bp = func.broadcast_to_param
        broadcast_target[func.output[0]] = (func.input[1], bp.axis)
        # we do not append node here because BroadcastTo should disappear
    elif func.type == "Add2":
        # Check if the second input is a brodcast target.
        bt = func.input[1]
        if bt in broadcast_target:
            merge_broadcast(n, func, bt, broadcast_target)
        nl.append(n)
    elif func.type == "Mul2":
        # Check if the second input is a brodcast target.
        bt = func.input[1]
        if bt in broadcast_target:
            merge_broadcast(n, func, bt, broadcast_target)
        nl.append(n)
    else:
        # Simply append node to list
        nl.append(n)
    return nl

def create_dim(val):
    """Create a dimension message for a given dimension"""
    dim = TensorShapeProto.Dimension()
    dim.dim_value = val
    return dim

def convert_parameter_shape(graph):
    """Convert the shape of some parameters so they fit ONNX's requirements.
    We do this as a post conversion because in the future we may be able to
    delete the whole conversion if NNabla's code gets changed"""
    batch_norm_constants = []
    for n in graph.node:
        if n.op_type == "BatchNormalization":
            # BatchNormalization in ONNX requires the scale, bias, mean, and variance input to be
            # one dimensional (https://github.com/onnx/onnx/blob/master/docs/Operators.md#batchnormalization).
            # However in NNabla these input must have a specific shape that matches the input shape.
            # For example if the input shape is (1,3,3,3), the above variables must have the shape (1,3,1,1) and not (3).
            # (1,3,1,1) is actually the same as a one-dimensional tensor of size 3,
            # but NNabla's check currently does not allow this.
            # Thus, we convert the shape of the above input so we can pass ONNX's check.
            # If NNabla or ONNX lightens the requirements, we should be able to remove this conversion.
            batch_norm_constants.extend(n.input[1:5])  # copy all input names for scale, bias, mean, variance

    # This loop should be fairly slow since we loop through all variables and parameters per constant
    for c in batch_norm_constants:
        # Reshape all BatchNormalization constant inputs assuming the size is (1,size,1,1)
        for i in graph.initializer:
            if i.name == c:
                size = i.dims
                if not (len(size) == 4 and
                        size[0] == 1 and size[2] == 1 and size[3] == 1):
                    raise ValueError(
                            "beta, gamma, mean, and variance parameters"
                            "must have the shape of 1*C*1*1 in {}".format(n.op_type))
                chan = size[1]
                del i.dims[:]
                i.dims.extend([chan])
                break
        for i in graph.input:
            if i.name == c:
                size = i.type.tensor_type.shape.dim
                if not (len(size) == 4 and
                        size[0].dim_value == 1 and
                        size[2].dim_value == 1 and
                        size[3].dim_value == 1):
                    raise ValueError(
                            "beta, gamma, mean, and variance parameters"
                            "must have the shape of 1*C*1*1 in {}".format(n.op_type))
                chan = size[1].dim_value
                del i.type.tensor_type.shape.dim[:]
                i.type.tensor_type.shape.dim.extend([create_dim(chan)])
                break


def get_tensor_type(name, type_dict):
    if name in type_dict:
        return type_dict[name]
    else:
        # Default tensor type to float
        return TensorProto.FLOAT


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

    # Store the names and type of all input/output
    # tensor that must have a type other than float.
    # If the input is in a parameter, it will be converted to that type
    input_types = {}
    output_types = {}
    # Store the input/output name of all BroadcastTo targets
    # so we can check if we can merge it to appropriate operators.
    broadcast_target = {}
    for f in net.function:
        nl = convert_to_nodes(f, net.variable, input_types, output_types, broadcast_target)
        graph.node.extend(nl)
    if len(broadcast_target) > 0:
        # If a broadcast target buffer is not used for any of the supported
        # operator's inputs, we throw an error.
        raise ValueError("BroadcastTo targets must be used in conjunction"
                         " with certain operators in order to get converted to ONNX")
    for param in nnp.parameter:
        init = graph.initializer.add()
        init.name = param.variable_name
        init.dims.extend(param.shape.dim)
        t = get_tensor_type(param.variable_name, input_types)
        init.data_type = t
        tensor_type_to_dtype = {
            TensorProto.FLOAT: np.float32,
            TensorProto.BOOL: np.bool
        }
        init.raw_data = np.array(param.data, dtype=tensor_type_to_dtype[t]).tostring()
        # init.float_data.extend(param.data)

    # Add all the constant parameters for all nodes
    # and the first node's input as input
    for iv in exe.data_variable:
        i = graph.input.add()
        i.name = iv.variable_name
        i.type.tensor_type.elem_type = get_tensor_type(iv.variable_name, input_types)
        dims = [create_dim(d) for d in var_dict[iv.variable_name].dim]
        i.type.tensor_type.shape.dim.extend(dims)
    for pv in exe.parameter_variable:
        p = graph.input.add()
        p.name = pv.variable_name
        p.type.tensor_type.elem_type = get_tensor_type(pv.variable_name, input_types)
        dims = [create_dim(d) for d in var_dict[pv.variable_name].dim]
        p.type.tensor_type.shape.dim.extend(dims)
    # Add only the final output of the graph as output
    for ov in exe.output_variable:
        o = graph.output.add()
        o.name = ov.variable_name
        o.type.tensor_type.elem_type = get_tensor_type(ov.variable_name, output_types)
        dims = [create_dim(d) for d in var_dict[ov.variable_name].dim]
        o.type.tensor_type.shape.dim.extend(dims)
    convert_parameter_shape(graph)


def nnp_model_to_onnx_protobuf(nnp):
    mp = ModelProto()
    mp.ir_version = ONNX_IR_VERSION
    op0 = mp.opset_import.add()
    op0.version = ONNX_OPSET_VERSION
    op1 = mp.opset_import.add()
    op1.domain = ""  # empty string indicates ONNX domain
    op1.version = ONNX_OPSET_VERSION
    # nn_opset = mp.opset_import.add()
    # nn_opset.domain = NNABLA_DOMAIN
    # nn_opset.version = NNABLA_OPSET_VERSION
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
