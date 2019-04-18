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
from functools import partial
from nnabla.utils import nnabla_pb2
import numpy as np
try:
    from onnx import (ModelProto, TensorProto, AttributeProto)
except:
    print('ONNX import support disabled because onnx python package is not found.')
    print(' You may install onnx package with "pip install onnx".')

from .utils import *

# We default to concat the channel axis
# so the concat results match with caffe2
DEFAULT_CONCAT_AXIS = 1
DEFAULT_SOFTMAX_AXIS = 1
# ONNX does not have the concept of executors.
# We will add a single executor to NNP when converted from ONNX,
# and set a default name to it.
DEFAULT_EXECUTOR_NAME = "exec_0"


def add_value_info_as_variable(network, info):
    if not info.type.HasField("tensor_type"):  # accepting only tensor
        raise ValueError("Only TensorProto is allowed as ValueInfoProto's type for info.name (Got {})"
                         .format(info.name, info.type))
    t = info.type.tensor_type
    v = network.variable.add()
    v.name = info.name
    v.shape.dim.extend([x.dim_value for x in t.shape.dim])
    return v


def add_value_info_as_parameter(network, info):
    v = add_value_info_as_variable(network, info)
    v.type = "Parameter"
    v.initializer.type = "Constant"
    v.initializer.multiplier = 1.0
    return v


def add_value_info_as_buffer(network, info):
    v = add_value_info_as_variable(network, info)
    v.type = "Buffer"
    return v


def set_kernel_parameter_and_add_padding(node, kp,
                                         pad_mode, pad_val,
                                         base_name, func_counter):
    """Set kernel related parameters(strides, pads, kernel_shape) to the given
    parameter. This function also generates a padding function if we need a
    seperate pad function for asymmetry padding.
    """
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
        elif attr.name == "count_include_pad":
            if attr.type != AttributeProto.INT:
                raise ValueError("Only INT is supported for count_include_pad in {} op_type"
                                 .format(node.op_type))
            kp.including_pad = bool(attr.i)
        else:
            raise ValueError("Unsupported attribute {} was specified at {}"
                             .format(attr.name, node.op_type))
    # NNabla requires for the dimensions of strides, pads, kernels to match.
    # We align the dimensions for all three attributes to the shortest one.
    dim = min(dims)
    padf = None
    if strides:
        kp.stride.dim.extend(strides[:])
    if pads:
        padval = []
        asymmetry = check_padding(pads, dim, padval)
        if asymmetry:
            # Add a separate padding function for
            # asymmetry padding
            input = node.input[0]
            padded = input+"_pad"
            pad_width = rearrange_pads(pads)
            padf = generate_pad(node.name, input, padded,
                                pad_mode, pad_width, pad_val,
                                base_name, func_counter)
        kp.pad.dim.extend(padval)
    else:
        # In case we don't have padding set,
        # we set zero padding just in case NNabla does not set the
        # default padding values correctly (such as in AveragePooling).
        # This code should not be needed
        # if NNabla handles default values correctly.
        kp.pad.dim.extend([0]*dim)
    if kernel:
        kp.kernel.dim.extend(kernel[:])
    return padf


def update_function_counter(func_type, func_counter, count):
    # Store the current usage count
    func_counter[func_type] = count+1


def generate_function_name(func_type, base_name, node_name, func_counter):
    # We are going to generate a name by counting
    # how many times a function was used.
    # (or we might use some kind of random number and hash it)
    count = 0
    if func_type in func_counter:
        # This function has been used already.
        # Get the current count
        count = func_counter[func_type]
    if node_name:
        # Include the node's name if it was specified.
        return "{}/{}/{}_{}".format(base_name, node_name, func_type, count), count
    else:
        return "{}/{}_{}".format(base_name, func_type, count), count


def set_function_name(func, node_name, base_name, func_counter):
    """Set a sufficient name for the function"""
    # NNabla requires each function to have a unique name
    # so we generate one here.
    func.name, count = generate_function_name(func.type, base_name, node_name,
                                              func_counter)
    update_function_counter(func.type, func_counter, count)


def generate_transpose(node_name, in_name, out_name, axes, base_name, func_counter):
    """Generate a Transpose operator to transpose the specified buffer.
    """
    trans = nnabla_pb2.Function()
    trans.type = "Transpose"
    set_function_name(trans, node_name, base_name, func_counter)
    trans.input.extend([in_name])
    trans.output.extend([out_name])
    tp = trans.transpose_param
    tp.axes.extend(axes)
    return trans


def generate_broadcast_to(node_name, x, y, out_name, axis, base_name, func_counter):
    """Generate a BroadcastTo operator to brodcast specified buffer"""
    bt = nnabla_pb2.Function()
    bt.type = "BroadcastTo"
    set_function_name(bt, node_name, base_name, func_counter)
    bt.input.extend([x, y])
    bt.output.extend([out_name])
    btp = bt.broadcast_to_param
    btp.axis = axis
    return bt


def generate_unary(func_name, node_name, x,
                   out_name, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = func_name
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    return func


def generate_sum(node_name, x, out_name,
                 axis, keepdims, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "Sum"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    sp = func.sum_param
    sp.axes.extend([axis])
    sp.keep_dims = keepdims
    return func


def generate_minimum_scalar(node_name, x, out_name,
                            val, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "MinimumScalar"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    msp = func.minimum_scalar_param
    msp.val = val
    return func


def generate_maximum_scalar(node_name, x, out_name,
                            val, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "MaximumScalar"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    msp = func.maximum_scalar_param
    msp.val = val
    return func


def generate_add_scalar(node_name, x, out_name,
                        val, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "AddScalar"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    asp = func.add_scalar_param
    asp.val = val
    return func


def generate_pow_scalar(node_name, x, out_name,
                        val, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "PowScalar"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    psp = func.pow_scalar_param
    psp.val = val
    return func


def generate_mul_scalar(node_name, x, out_name,
                        val, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "MulScalar"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    msp = func.mul_scalar_param
    msp.val = val
    return func


def generate_sum_pooling(node_name, x, out_name,
                         kernel, stride, ignore_border, pad,
                         base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "SumPooling"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    spp = func.sum_pooling_param
    spp.kernel.dim.extend(kernel)
    spp.stride.dim.extend(stride)
    spp.ignore_border = ignore_border
    spp.pad.dim.extend(pad)
    return func


def generate_pad(node_name, x, out_name,
                 mode, pad_width, constant_val,
                 base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "Pad"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    pp = func.pad_param
    pp.mode = mode
    pp.pad_width.extend(pad_width)
    pp.constant_value = constant_val
    return func


def generate_reshape(node_name, x, out_name,
                     output_shape, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "Reshape"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    rp = func.reshape_param
    rp.shape.dim.extend(output_shape)
    return func


def set_reduction_attrs(p, node):
    p.keep_dims = True  # keep_dims is default True for ONNX
    for attr in node.attribute:
        if attr.name == "axes":
            if attr.type != AttributeProto.INTS:
                raise ValueError(
                    "Only INTS is supported for axes in {} op_type".format(node.op_type))
            p.axes.extend(attr.ints)
        elif attr.name == "keepdims":
            if attr.type != AttributeProto.INT:
                raise ValueError(
                    "Only INT is supported for keepdims in {} op_type".format(node.op_type))
            p.keep_dims = bool(attr.i)
        else:
            raise ValueError("Unsupported attribute {} was specified at {}"
                             .format(attr.name, node.op_type))


def check_padding(pads, dim, padval):
    """Check each padding start/end value
    and set the sufficient pad value.
    If we have asymmetry padding, we will return
    True to indicate the need for a separate padding function"""
    asymmetry = False
    for i in range(dim):
        s = pads[i]
        e = pads[i+dim]
        if s == e:
            padval.append(s)
        else:
            asymmetry = True
            # We must add a separate pad function for asymmetry padding.
            # Since the pad function will do all the padding,
            # we will remove all padding here.
            del padval[:]
            padval.extend([0]*dim)
            break
    return asymmetry


def rearrange_pads(pads):
    """ Interleave pad values to match NNabla format
    (S0,S1,E0,E1) => (S0,E0,S1,E1)"""
    half = len(pads)//2
    starts = pads[:half]
    ends = pads[half:]
    return [j for i in zip(starts, ends) for j in i]


def convert_parameter_shape(pb):
    """Convert the shape of some parameters so they fit NNabla's requirements.
    We do this as a post conversion because in the future we may be able to
    delete the whole conversion if NNabla's code gets changed"""
    if len(pb.network) != 1:
        raise ValueError(
            "NNP with more then a single network is currently not supported")
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
            # copy all input names for scale, bias, mean, variance
            batch_norm_constants.extend(f.input[1:5])

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


def add_tensor_as_parameter(pb, tensor):
    """Add given tensor as a parameter"""
    p = pb.parameter.add()
    p.variable_name = tensor.name
    p.shape.dim.extend(tensor.dims)
    if tensor.data_type == TensorProto.FLOAT:
        # convert raw bytestream to floating points
        if tensor.raw_data:
            p.data.extend(np.fromstring(tensor.raw_data, dtype=np.float32))
        elif len(tensor.float_data) > 0:
            p.data.extend(tensor.float_data)
        else:
            raise ValueError("float data not found for {}".format(tensor.name))
    elif tensor.data_type == TensorProto.INT32:
        # convert raw bytestream to integer
        if tensor.raw_data:
            p.data.extend(np.fromstring(tensor.raw_data, dtype=np.int32))
        elif len(tensor.int32_data) > 0:
            p.data.extend(tensor.int32_data)
        else:
            raise ValueError("int32 data not found for {}".format(tensor.name))
    elif tensor.data_type == TensorProto.INT64:
        # convert raw bytestream to integer
        if tensor.raw_data:
            p.data.extend(np.fromstring(tensor.raw_data, dtype=np.int64))
        elif len(tensor.int64_data) > 0:
            p.data.extend(tensor.int64_data)
        else:
            raise ValueError("int64 data not found for {}".format(tensor.name))
    elif tensor.data_type == TensorProto.BOOL:
        if tensor.raw_data:
            p.data.extend(np.fromstring(tensor.raw_data, dtype=np.bool))
        else:
            raise ValueError("bool data not found for {}".format(tensor.name))

    else:
        raise ValueError("Unsupported tensor data type for {}: {}"
                         .format(tensor.name, tensor.data_type))
    p.need_grad = False


class OnnxImporter:
    def __init__(self, file_path=''):
        self._file_path = file_path

        # We use an OrderedDict and not a set
        # to preserve order
        self._param_vars = OrderedDict()  # Dictionary for input parameters.
        self._all_vars = OrderedDict()  # Dictionary for all variables
        self._param_list = []  # list of parameter variables
        self._merged_inputs = []  # list of input buffers that was merged to a function
        self._removed_outputs = []  # list of output buffers that was removed
        self._func_counter = {}  # a counter for all functions
        self._shape_output = {}  # The shape of the output that all functions

        # Dictionary used to convert ONNX op_type to processing method of NNabla function
        # opset_6 default op table
        self.table_op_set_6 = {
            "Dropout": partial(self.Dropout, 6),
            "Softmax": self.Softmax,
            "BatchNormalization": self.BatchNormalization,
            "Reshape": self.Reshape,
            "Transpose": self.Transpose,
            "Abs": partial(self.GeneralOperator, 'Abs'),
            "Sigmoid": partial(self.GeneralOperator, 'Sigmoid'),
            "Tanh": partial(self.GeneralOperator, 'Tanh'),
            "Log": partial(self.GeneralOperator, 'Log'),
            "Less": partial(self.BroadcastOperator, 'Less'),
            "Greater": partial(self.BroadcastOperator, 'Greater'),
            "Equal": partial(self.BroadcastOperator, 'Equal'),
            "Exp": partial(self.GeneralOperator, 'Exp'),
            "Identity": partial(self.GeneralOperator, 'Identity'),
            "Pad": self.Pad,
            "Relu": partial(self.ReLU, 'ReLU'),
            "PRelu": partial(self.ReLU, 'PReLU'),
            "Concat": self.Concatenate,
            "Conv": self.Convolution,
            "GlobalAveragePool": partial(self.BasePooling, 'GlobalAveragePooling'),
            "MaxPool": partial(self.BasePooling, 'MaxPooling'),
            "AveragePool": partial(self.BasePooling, 'AveragePooling'),
            "Sum": partial(self.BroadcastOperator_9, 'Add2'),
            "Gemm": self.Affine,
            "Add": partial(self.BroadcastOperator, 'Add2'),
            "Mul": partial(self.BroadcastOperator, 'Mul2'),
            "Div": partial(self.BroadcastOperator, 'Div2'),
            "Pow": partial(self.BroadcastOperator, 'Pow2'),
            "Sub": partial(self.BroadcastOperator, 'Sub2'),
            "MatMul": self.BatchMatmul,
            "LeakyRelu": partial(self.ELU, 'LeakyReLU'),
            "Not": partial(self.GeneralOperator, 'LogicalNot'),
            "Elu": partial(self.ELU, 'ELU'),
            "Selu": partial(self.ELU, 'SELU'),
            "ReduceSum": partial(self.ReduceScalar, 'Sum'),
            "ReduceMean": partial(self.ReduceScalar, 'Mean'),
            "ReduceMin": partial(self.ReduceScalar, 'Min'),
            "ReduceMax": partial(self.ReduceScalar, 'Max'),
            "ReduceProd": partial(self.ReduceScalar, 'Prod'),
            "And": partial(self.BroadcastOperator, 'LogicalAnd'),
            "Or": partial(self.BroadcastOperator, 'LogicalOr'),
            "Xor": partial(self.BroadcastOperator, 'LogicalXor'),
            "Min": partial(self.BroadcastOperator_9, 'Minimum2'),
            "Max": partial(self.BroadcastOperator_9, 'Maximum2'),
            "Reciprocal": partial(self.ElementWiseScalar, 'RDivScalar'),
            "Neg": partial(self.ElementWiseScalar, 'MulScalar'),
            "LogSoftmax": self.LogSoftmax,
            "Softplus": self.Softplus,
            "Softsign": self.Softsign,
            "LRN": self.LRN,
            # Clip gets converted to Identity, MaxScalar or MinScalar
            # or both, depending on the attributes.
            # We set a temporary name here
            "Clip": self.Clip,
            # Constant does not get converted to a function
            # but we list it here so we can accept it
            "Constant": self.Constant,
            "Unsqueeze": self.Unsqueeze
        }

        # opset_9 table
        self.table_op_set_9 = {
            "Dropout": partial(self.Dropout, 9),
            "Less": partial(self.BroadcastOperator_9, 'Less'),
            "Greater": partial(self.BroadcastOperator_9, 'Greater'),
            "Equal": partial(self.BroadcastOperator_9, 'Equal'),
            "Add": partial(self.BroadcastOperator_9, 'Add2'),
            "Mul": partial(self.BroadcastOperator_9, 'Mul2'),
            "Div": partial(self.BroadcastOperator_9, 'Div2'),
            "Pow": partial(self.BroadcastOperator_9, 'Pow2'),
            "Sub": partial(self.BroadcastOperator_9, 'Sub2'),
            "And": partial(self.BroadcastOperator_9, 'LogicalAnd'),
            "Or": partial(self.BroadcastOperator_9, 'LogicalOr'),
            "Xor": partial(self.BroadcastOperator_9, 'LogicalXor'),
        }
        self.table_op_set_9 = dict(self.table_op_set_6, **self.table_op_set_9)

        # Currently, we only planed to support opset 6 and opset 9.
        # More planes will be added later to support more opset versions.
        self.opver_impl_map = {
            "6": self.table_op_set_6,
            "9": self.table_op_set_9
        }

    def get_onnx_graph_info(self):
        model_proto = ModelProto()
        with open(self._file_path, "rb") as f:
            model_proto.ParseFromString(f.read())
        self._ir_version = model_proto.ir_version
        self._graph = model_proto.graph
        self._opset_import = model_proto.opset_import

    def check_domain(self, domain):
        # We do not allow any operator from an unknown domain
        if not (domain == '' or domain == NNABLA_DOMAIN):
            raise ValueError(
                "Unsupported operator from domain {} was found".format(domain))

    def get_func_input_shape(self, input_name):
        input_shape = []
        if input_name in self._shape_output:
            input_shape.extend(self._shape_output[input_name])
        else:
            for i in self._graph.input:
                if i.name == input_name:
                    t = i.type.tensor_type
                    input_shape = [x.dim_value for x in t.shape.dim]
        if not input_shape:
            raise ValueError(
                "The shape of {} was not found".format(input_name))
        return input_shape

    def generate_default_function(self, func_name, n):
        func = nnabla_pb2.Function()
        func.type = func_name
        set_function_name(func, n.name, self._graph.name, self._func_counter)
        func.input.extend(n.input)
        func.output.extend(n.output)
        return func

    def Convolution(self, func_list, n):
        func = self.generate_default_function("Convolution", n)
        cp = func.convolution_param
        # We shouldn't need these default settings
        # since NNabla will set these for us
        cp.base_axis = 1
        cp.group = 1
        dims = []
        pads = []
        strides = []
        dilations = []
        for attr in n.attribute:
            if attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for pads in Conv op_type")
                pads.extend(attr.ints)
                dims.append(len(pads))
            elif attr.name == "strides":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for strides in Conv op_type")
                strides.extend(attr.ints)
                dims.append(len(strides))
            elif attr.name == "dilations":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for dilations in Conv op_type")
                dilations.extend(attr.ints)
                dims.append(len(dilations))
            elif attr.name == "group":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for group in Conv op_type")
                cp.group = attr.i
            elif attr.name == "kernel_shape":
                # We do not set 'kernel_shape' to NNabla
                # since NNabla doesn't have a parameter for it
                # (it will be inferred from weight input)
                pass
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        # NNabla requires for the dimensions of strides, pads, dilations to match.
        # We align the dimensions for all three attributes to the shortest one
        dim = min(dims)
        if strides:
            cp.stride.dim.extend(strides[:])
        if pads:
            padval = []
            asymmetry = check_padding(pads, dim, padval)
            if asymmetry:
                # Add a separate padding function for
                # asymmetry padding
                input = n.input[0]
                padded = input+"_pad"
                pad_width = rearrange_pads(pads)
                padf = generate_pad(n.name, input, padded,
                                    "replicate", pad_width, 0,
                                    self._graph.name, self._func_counter)
                output_shape = []
                input_shape = self.get_func_input_shape(input)
                s = len(pad_width) // 2
                shape = input_shape[-s:]
                for i in range(s):
                    shape[i] += pad_width[2 * i]
                    shape[i] += pad_width[2 * i + 1]
                output_shape.extend(input_shape[:-s])
                output_shape.extend(shape)
                self._shape_output[padded] = output_shape
                func_list.append(padf)
                # Rewire input to the padded version
                del func.input[:]
                func.input.extend(padded)
            cp.pad.dim.extend(padval)
        else:
            # Set default values.
            # Do we really need this? (Default value should be set by NNabla)
            cp.pad.dim.extend([0]*dim)
        if dilations:
            cp.dilation.dim.extend(dilations[:])
        else:
            # Set default values.
            # Do we really need this? (Default value should be set by NNabla)
            cp.dilation.dim.extend([1]*dim)
        weight_shape = []
        output_shape = []
        input_shape = self.get_func_input_shape(func.input[0])
        for i in self._graph.initializer:
            if i.name == func.input[1]:
                weight_shape = i.dims
        output_shape = input_shape[:1]
        output_shape.append(weight_shape[0])
        for index in range(dim):
            d = cp.dilation.dim[index]
            p = cp.pad.dim[index]
            s = cp.stride.dim[index]
            w = weight_shape[2+index]
            i = input_shape[2+index]
            k = d * (w - 1) + 1
            o = (i + 2 * p - k) // s + 1
            output_shape.append(o)
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def BatchNormalization(self, func_list, n):
        func = self.generate_default_function("BatchNormalization", n)
        # We need to rearrange the input data order.
        # ONNX BatchNormalization input order: X, scale, bias, mean, variance
        # NNabla BatchNormalization input order: X, beta, gamma, mean, variance
        nnp_order = [0, 2, 1, 3, 4]
        if len(n.input) != len(nnp_order):
            raise ValueError(
                "The number of BatchNormalization input must be {}".format(len(nnp_order)))
        nnp_input = [n.input[i] for i in nnp_order]
        del func.input[:]
        func.input.extend(nnp_input)
        bnp = func.batch_normalization_param
        # Set default axis.
        # We shouldn't need this if the default is set properly
        bnp.axes.extend([1])
        for attr in n.attribute:
            if attr.name == "is_test":
                pass
            elif attr.name == "epsilon":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for epsilon in BatchNormalization op_type")
                bnp.eps = attr.f
            elif attr.name == "momentum":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for momentum in BatchNormalization op_type")
                bnp.decay_rate = attr.f
            elif attr.name == "consumed_inputs":
                # BatchNormalization-1 has this field.
                # Since NNabla does not need this, we ignore it
                pass
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        if len(n.output) > 1:
            bnp.batch_stat = True
        else:
            bnp.batch_stat = False
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

    def ReLU(self, func_name, func_list, n):
        func = self.generate_default_function(func_name, n)
        if func_name == "PReLU":
            pp = func.prelu_param
            # ONNX PRelu defaults to the Channel axis,
            # so we set the channel axis (1) here.
            # This should be the same for NNabla
            # buf currently it defaults to 0
            # so we explicitly set 1 here.
            pp.base_axis = 1
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

    def Reshape(self, func_list, n):
        func = self.generate_default_function("Reshape", n)
        rp = func.reshape_param
        shape_found = False
        for attr in n.attribute:
            if attr.name == "shape":
                # Shape comes as attribute for Reshape-1
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for shape in {} op_type".format(n.op_type))
                rp.shape.dim.extend(attr.ints)
                shape_found = True
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        if len(func.input) == 2:
            # Shape comes as input for Reshape-5.
            # NNabla reshape excepts a single input (data),
            # while Reshape-5 will have two inputs (data, shape),
            # so we convert the shape input to a parameter
            shape_input = func.input[1]
            # look for the initializer for matching input
            for init in self._graph.initializer:
                if init.name == shape_input:
                    if init.data_type != TensorProto.INT64:
                        raise ValueError(
                            "Only INT64 is supported for shape in {} op_type".format(n.op_type))
                    # copy shape size from initializer
                    if init.raw_data:
                        rp.shape.dim.extend(np.fromstring(
                            init.raw_data, dtype=np.int64))
                    elif init.int64_data:
                        rp.shape.dim.extend(init.int64_data)
                    shape_found = True
                    break
            # stored the merged input so we can ignore it later
            self._merged_inputs.append(shape_input)
            del func.input[1]
        if not shape_found:
            raise ValueError(
                "Shape information was not found in {} op_type".format(n.op_type))
        self._shape_output[func.output[0]] = rp.shape.dim
        func_list.append(func)

    def Transpose(self, func_list, n):
        func = self.generate_default_function("Transpose", n)
        tp = func.transpose_param
        for attr in n.attribute:
            if attr.name == "perm":
                # perm has the same meaning for ONNX and NNabla
                # so we simply copy the parameter
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for perm in {} op_type".format(n.op_type))
                tp.axes.extend(attr.ints)
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        input_shape = self.get_func_input_shape(func.input[0])
        output_shape = []
        for i in range(len(input_shape)):
            index = tp.axes[i]
            output_shape.append(input_shape[index])
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def BasePooling(self, func_name, func_list, n):
        if func_name == 'AveragePooling':
            func = self.generate_default_function("AveragePooling", n)
            kp = func.average_pooling_param
            padf = set_kernel_parameter_and_add_padding(n, kp,
                                                        "replicate", 0,
                                                        self._graph.name, self._func_counter)
        elif func_name == 'MaxPooling':
            func = self.generate_default_function("MaxPooling", n)
            kp = func.max_pooling_param
            # We simulate replicate mode by padding with negative infinite
            padf = set_kernel_parameter_and_add_padding(n, kp,
                                                        "constant", -np.inf,
                                                        self._graph.name, self._func_counter)
        elif func_name == 'GlobalAveragePooling':
            func = self.generate_default_function("GlobalAveragePooling", n)
            input_shape = self.get_func_input_shape(func.input[0])
            output_shape = []
            output_shape.extend(input_shape[:2])
            output_shape.extend([1, 1])
            self._shape_output[func.output[0]] = output_shape
            func_list.append(func)
            return
        if padf:
            # append a pad function if we need asymmetry padding
            input_shape = self.get_func_input_shape(padf.input[0])
            output_shape = []
            s = len(padf.pad_param.pad_width) // 2
            shape = input_shape[-s:]
            for i in range(s):
                shape[i] += padf.pad_param.pad_width[2 * i]
                shape[i] += padf.pad_param.pad_width[2 * i + 1]
            output_shape.extend(input_shape[:-s])
            output_shape.extend(shape)
            self._shape_output[padf.output[0]] = output_shape
            func_list.append(padf)
            # Rewire input to the padded version
            del func.input[:]
            func.input.extend(padf.output)
        kp.ignore_border = True
        output_shape = []
        input_shape = self.get_func_input_shape(func.input[0])
        s = len(input_shape) - len(kp.kernel.dim)
        shape = input_shape[s:]
        for i in range(len(shape)):
            shape[i] += 2 * kp.pad.dim[i]
            shape[i] = (shape[i] - kp.kernel.dim[i]) // kp.stride.dim[i] + 1
        for i in range(len(input_shape)):
            if i < s:
                output_shape.append(input_shape[i])
            else:
                output_shape.append(shape[i - s])
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Concatenate(self, func_list, n):
        # Concat axis was not required for Concat-1 (it is required from Concat-4),
        # so the default axis depended on which backend we use.
        # Since we are comparing with caffe2, we are
        # defaulting to the channel axis if the axis is not specified.
        # https://github.com/onnx/onnx/issues/374
        func = self.generate_default_function("Concatenate", n)
        func.concatenate_param.axis = DEFAULT_CONCAT_AXIS
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Axis type must be a single integer")
                # The axis was specified so we use it
                func.concatenate_param.axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        input_shape = self.get_func_input_shape(func.input[0])
        axis_ = func.concatenate_param.axis
        for i in range(len(func.input)):
            if i >= 1:
                input_shape[axis_] += self.get_func_input_shape(func.input[i])[
                                                                axis_]
        self._shape_output[func.output[0]] = input_shape
        func_list.append(func)

    def BroadcastOperator(self, func_name, func_list, n):
        """Converts a broadcasting operator to a composite with BroadcastTo"""
        broadcasting = False
        broadcast_axis = -1
        func = self.generate_default_function(func_name, n)
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for axis in {} op_type".format(n.op_type))
                broadcast_axis = attr.i
            elif attr.name == "broadcast":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for broadcast in {} op_type".format(n.op_type))
                if attr.i == 1:
                    broadcasting = True
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        if not broadcasting:
            input0_shape = self.get_func_input_shape(func.input[0])
            input1_shape = self.get_func_input_shape(func.input[1])
            output_shape = []
            for i in range(len(input0_shape)):
                output_shape.append(max(input0_shape[i], input1_shape[i]))
            self._shape_output[func.output[0]] = output_shape
            func_list.append(func)
            return
        # Create a BroadcastTo operator to broadcast input B
        b_idx = 1  # B is the second input
        broadcasted_postfix = "_broadcasted"
        input = n.input[:]
        bin = n.input[b_idx]
        bout = bin+broadcasted_postfix
        bt = generate_broadcast_to(n.name, n.input[0], bin, bout, broadcast_axis,
                                   self._graph.name, self._func_counter)
        self._shape_output[bout] = self.get_func_input_shape(n.input[0])
        func_list.append(bt)
        input[b_idx] = bout  # rewire input to broadcasted input
        # update input with the converted inputs
        del func.input[:]
        func.input.extend(input)
        self._shape_output[func.output[0]] = self._shape_output[bout]
        func_list.append(func)

    def BroadcastOperator_9(self, func_name, func_list, n):
        func = self.generate_default_function(func_name, n)
        # NNabla can only process two inputs for max/min/sum.
        # Check if this is fulfilled.
        if len(n.input) != 2:
            raise ValueError(
                "NNabla can only process Min/Max/Sum of two tensors")
        input0_shape = self.get_func_input_shape(func.input[0])
        input1_shape = self.get_func_input_shape(func.input[1])
        input0_dim = len(input0_shape)
        input1_dim = len(input1_shape)
        if input0_dim != input1_dim:
            if input0_dim == 0:
                del func.input[0]
                func.type = "Identity"
                self._shape_output[func.output[0]] = input1_shape
            elif input1_dim == 0:
                del func.input[1]
                func.type = "Identity"
                self._shape_output[func.output[0]] = input0_shape
            else:
                ndim = max(input0_dim, input1_dim)
                shape = []
                output_shape = []
                if input0_dim < ndim:
                    shape.extend(list(np.ones(ndim-input0_dim, dtype=np.int)))
                    shape.extend(input0_shape)
                    out = n.input[0]+"_shape"
                    rp = generate_reshape(n.name, n.input[0], out, shape,
                                          self._graph.name, self._func_counter)
                    self._shape_output[out] = shape
                    func_list.append(rp)
                    func.input[0] = out
                    output_shape = [max(shape[i], input1_shape[i])
                                    for i in range(ndim)]
                elif input1_dim < ndim:
                    shape.extend(list(np.ones(ndim-input1_dim, dtype=np.int)))
                    shape.extend(input1_shape)
                    out = n.input[1]+"_shape"
                    rp = generate_reshape(n.name, n.input[1], out, shape,
                                          self._graph.name, self._func_counter)
                    self._shape_output[out] = shape
                    func_list.append(rp)
                    func.input[1] = out
                    output_shape = [max(shape[i], input0_shape[i])
                                    for i in range(ndim)]
                self._shape_output[func.output[0]] = output_shape
            func_list.append(func)
        else:
            output_shape = []
            for i in range(len(input0_shape)):
                output_shape.append(max(input0_shape[i], input1_shape[i]))
            self._shape_output[func.output[0]] = output_shape
            func_list.append(func)

    def Affine(self, func_list, n):
        func = self.generate_default_function("Affine", n)
        ap = func.affine_param
        ap.base_axis = 1
        transposed_postfix = "_transposed"
        input = n.input[:]
        # Check the dimension of all inputs
        x = input[0]
        weight = input[1]
        bias = input[2]
        for init in self._graph.initializer:
            if init.name == x or init.name == weight:
                # must be two dimensional
                if len(init.dims) != 2:
                    raise ValueError(
                        "Only two dimensional input is currently supported for Gemm input tensor A and B ({})".format(init.name))
            elif init.name == bias:
                # Must be one dimensional
                if len(init.dims) != 1:
                    raise ValueError(
                        "Only one dimensional input is currently supported for Gemm input tensor C ({})".format(init.name))

        # Switch H and W for transpose
        # We assume the buffer is two dimensional.
        axes = [1, 0]
        for attr in n.attribute:
            if attr.name == "transA":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for transA in {} op_type".format(n.op_type))
                # We need to transpose the input weight beforehand
                # since NNabla does not support transpose with Affine.
                # Add a new intermediate buffer for transposition,
                # and rewire the buffer as input.
                ain = n.input[0]
                aout = ain+transposed_postfix
                transA = generate_transpose(
                    n.name, ain, aout, axes, self._graph.name, self._func_counter)
                input_shape = self.get_func_input_shape(ain)
                output_shape = []
                for i in range(len(input_shape)):
                    index = axes[i]
                    output_shape.append(input_shape[index])
                self._shape_output[aout] = output_shape
                func_list.append(transA)
                input[0] = aout  # rewire input to transposed input
            elif attr.name == "transB":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for transB in {} op_type".format(n.op_type))
                # same as transA
                bin = n.input[1]
                bout = bin+transposed_postfix
                transB = generate_transpose(
                    n.name, bin, bout, axes, self._graph.name, self._func_counter)
                input_shape = self.get_func_input_shape(bin)
                output_shape = []
                for i in range(len(input_shape)):
                    index = axes[i]
                    output_shape.append(input_shape[index])
                self._shape_output[bout] = output_shape
                func_list.append(transB)
                input[1] = bout  # rewire input to transposed input
            elif attr.name == "broadcast":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for broadcast in {} op_type".format(n.op_type))
                # Affine broadcasts Bias vector automatically if the bias is one dimension
                # so we don't have to do anything here
                pass
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        # update Gemm input with the converted inputs
        del func.input[:]
        func.input.extend(input)
        input_shape = self.get_func_input_shape(func.input[0])
        weight_shape = self.get_func_input_shape(func.input[1])
        output_shape = []
        output_shape.append(input_shape[0])
        output_shape.extend(weight_shape[1:])
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Softmax(self, func_list, n):
        func = self.generate_default_function("Softmax", n)
        logger.warning(SOFTMAX_WARNING)
        # default to channel axis
        func.softmax_param.axis = DEFAULT_SOFTMAX_AXIS
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Softmax axis must be a single integer")
                func.softmax_param.axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        input_shape = self.get_func_input_shape(func.input[0])
        self._shape_output[func.output[0]] = input_shape
        func_list.append(func)

    def Pad(self, func_list, n):
        func = self.generate_default_function("Pad", n)
        mode = "constant"
        pads = []
        value = 0
        for attr in n.attribute:
            if attr.name == "mode":
                if attr.type != AttributeProto.STRING:
                    raise ValueError("mode must be a string for Op: {}"
                                     .format(n.op_type))
                mode = attr.s.decode("utf-8")
            elif attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("pads must be a list of ints for Op: {}"
                                     .format(n.op_type))
                pads = attr.ints
            elif attr.name == "value":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("value must be a single float for Op: {}"
                                     .format(n.op_type))
                value = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        if len(pads) == 0:
            raise ValueError("Required attribute pads not found for {}"
                             .format(n.op_type))
        mode_conv = {
            "constant": "constant",
            "edge": "replicate",
            "reflect": "reflect"
        }
        pp = func.pad_param
        pp.mode = mode_conv[mode]
        pp.constant_value = value
        pw = rearrange_pads(pads)
        pp.pad_width.extend(pw)
        output_shape = []
        input_shape = self.get_func_input_shape(func.input[0])
        s = len(pp.pad_width) // 2
        shape = input_shape[-s:]
        for i in range(s):
            shape[i] += pp.pad_width[2 * i]
            shape[i] += pp.pad_width[2 * i + 1]
        output_shape.extend(input_shape[:-s])
        output_shape.extend(shape)
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Dropout(self, opset, func_list, n):
        func = self.generate_default_function("Dropout", n)
        if len(n.output) > 1:
            # An ONNX Dropout node may have two outputs (result + mak)
            # while a NNabla Dropout/Identity only allows a single output.
            # We will drop the mask output (which should be the second one).
            # This may result in a broken network (if the mask output was used later)
            # so we show a warning here
            logger.warning("Dropout's mask output {} will be removed"
                           " since NNabla does not produce mask output".format(n.output[1]))
            self._removed_outputs.append(n.output[1])
            del func.output[:]
            func.output.extend([n.output[0]])
        # Dropout requires a ratio to be set
        for attr in n.attribute:
            if attr.name == "is_test":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Dropout is_test must be a single integer")
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
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        if opset >= 7:
            # The ONNX official document does not explicitly write
            # the distinction test mode and training mode.
            # Refer to the implementation of caffe2,
            # The version of opset >= 7 is processed in test mode.
            func.ClearField("dropout_param")
            func.type = "Identity"
        input_shape = self.get_func_input_shape(func.input[0])
        self._shape_output[func.output[0]] = input_shape
        func_list.append(func)

    def LRN(self, func_list, n):
        func = self.generate_default_function("Div2", n)
        # Gather attributes.
        # The following are default values for ONNX
        alpha = 1e-4
        beta = 0.75
        bias = 1.0
        size = -1
        for attr in n.attribute:
            if attr.name == "alpha":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("alpha must be a single float for Op: {}"
                                     .format(n.op_type))
                alpha = attr.f
            elif attr.name == "beta":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("beta must be a single float for Op: {}"
                                     .format(n.op_type))
                beta = attr.f
            elif attr.name == "bias":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("bias must be a single float for Op: {}"
                                     .format(n.op_type))
                bias = attr.f
            elif attr.name == "size":
                if attr.type != AttributeProto.INT:
                    raise ValueError("size must be a single integer for Op: {}"
                                     .format(n.op_type))
                size = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        if size < 0:
            raise ValueError("Size is required for {}"
                             .format(n.op_type))
        elif (size % 2) == 0:
            raise ValueError("Only size with odd values is "
                             "currently supported for {}"
                             .format(n.op_type))

        # Convert to PowScalar+Transpose+SumPooling+Transpose+
        # MulScalar+AddScalar+PowScalar
        pow0_in = n.input[0]
        pow0_out = pow0_in+"_pow0"
        pow0 = generate_pow_scalar(n.name, pow0_in, pow0_out,
                                   2, self._graph.name, self._func_counter)
        self._shape_output[pow0_out] = self.get_func_input_shape(pow0_in)
        func_list.append(pow0)
        # Transpose the channel axis so we can sumpool along the channels
        # We are assuming 4D input
        trans0_out = pow0_out+"_trans0"
        axes = [0, 2, 3, 1]
        trans0 = generate_transpose(n.name, pow0_out, trans0_out,
                                    axes, self._graph.name, self._func_counter)
        input_shape = self._shape_output[pow0_out]
        output_shape = []
        for i in range(len(input_shape)):
            index = axes[i]
            output_shape.append(input_shape[index])
        self._shape_output[trans0_out] = output_shape
        func_list.append(trans0)
        # SumPool along channels.
        padval = (size - 1)//2
        sp_out = trans0_out+"_sp"
        sump = generate_sum_pooling(n.name, trans0_out, sp_out,
                                    [1, size], [1, 1], True, [0, padval],
                                    self._graph.name, self._func_counter)
        kp = sump.sum_pooling_param
        output_shape = []
        input_shape = self._shape_output[trans0_out]
        s = len(input_shape) - len(kp.kernel.dim)
        shape = input_shape[s:]
        for i in range(len(shape)):
            shape[i] += 2 * kp.pad.dim[i]
            shape[i] = (shape[i] - kp.kernel.dim[i]) // kp.stride.dim[i] + 1
        for i in range(len(input_shape)):
            if i < s:
                output_shape.append(input_shape[i])
            else:
                output_shape.append(shape[i - s])
        self._shape_output[sp_out] = output_shape
        func_list.append(sump)
        # Transpose back
        trans1_out = sp_out+"_trans1"
        axes = [0, 3, 1, 2]
        trans1 = generate_transpose(n.name, sp_out, trans1_out,
                                    axes, self._graph.name, self._func_counter)
        input_shape = self._shape_output[sp_out]
        output_shape = []
        for i in range(len(input_shape)):
            index = axes[i]
            output_shape.append(input_shape[index])
        self._shape_output[trans1_out] = output_shape
        func_list.append(trans1)
        # MulScalar
        muls_out = trans1_out+"_muls"
        muls = generate_mul_scalar(n.name, trans1_out, muls_out,
                                   alpha/size, self._graph.name, self._func_counter)
        self._shape_output[muls_out] = self._shape_output[trans1_out]
        func_list.append(muls)
        # AddScalar
        adds_out = muls_out+"_adds"
        adds = generate_add_scalar(n.name, muls_out, adds_out,
                                   bias, self._graph.name, self._func_counter)
        self._shape_output[adds_out] = self._shape_output[muls_out]
        func_list.append(adds)
        # PowScalar
        pow1_out = adds_out+"_pow1"
        pow1 = generate_pow_scalar(n.name, adds_out, pow1_out,
                                   beta, self._graph.name, self._func_counter)
        self._shape_output[pow1_out] = self._shape_output[adds_out]
        func_list.append(pow1)
        # rewire Div2 input to original input and PowScalar output
        del func.input[:]
        func.input.extend([pow0_in, pow1_out])
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

    def ReduceScalar(self, func_name, func_list, n):
        func = self.generate_default_function(func_name, n)
        if func_name == "Sum":
            func_param = func.sum_param
        elif func_name == "Mean":
            func_param = func.mean_param
        elif func_name == "Min":
            func_param = func.min_param
        elif func_name == "Max":
            func_param = func.max_param
        elif func_name == "Prod":
            func_param = func.prod_param
        else:
            func_param = None
        if func_param:
            set_reduction_attrs(func_param, n)
            output_shape = self.get_func_input_shape(func.input[0])
            for i in func_param.axes:
                if func_param.keep_dims:
                    output_shape[i] = 1
                else:
                    del output_shape[i]
            self._shape_output[func.output[0]] = output_shape
            func_list.append(func)

    def Constant(self, func_list, n):
        # Convert a Constant node as an input parameter and not a function
        assert len(n.output) == 1, "Constant output must be a single buffer"
        name = n.output[0]
        for attr in n.attribute:
            if attr.name == "value":
                if attr.type != AttributeProto.TENSOR:
                    raise ValueError(
                        "Only TENSOR is supported for value in {} op_type".format(n.op_type))
                t = attr.t
                if t is None:
                    raise ValueError(
                        "value attribute must be set for {}".format(n.op_type))
                t.name = name
                # add tensor as parameter
                add_tensor_as_parameter(self._pb, t)
                self._param_vars[t.name] = None
                # Add tensor as variable
                v = self._pb.network[0].variable.add()
                v.name = t.name
                v.shape.dim.extend(t.dims)
                v.type = "Parameter"
                v.initializer.type = "Constant"
                v.initializer.multiplier = 1.0
                self._param_list.append(v)
                self._shape_output[name] = t.dims
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        # We do not add any function to the list here
        # since the node is converted as a parameter

    def ELU(self, func_name, func_list, n):
        func = self.generate_default_function(func_name, n)
        if func_name == "LeakyReLU":
            func_param = func.leaky_relu_param
            func_param.alpha = 0.01  # alpha value defaults to 0.01 in ONNX
        elif func_name == "ELU":
            func_param = func.elu_param
            func_param.alpha = 1.0  # alpha value defaults to 1.0 in ONNX
        elif func_name == "SELU":
            func_param = func.selu_param
            func_param.alpha = 1.6732  # Default value for ONNX
            func_param.scale = 1.0507
        for attr in n.attribute:
            if attr.name == "alpha":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for alpha in {} op_type".format(n.op_type))
                func_param.alpha = attr.f
            elif attr.name == "gamma":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for gamma in {} op_type".format(n.op_type))
                func_param.scale = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

    def GeneralOperator(self, func_name, func_list, n):
        func = self.generate_default_function(func_name, n)
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

    def BatchMatmul(self, func_list, n):
        func = self.generate_default_function("BatchMatmul", n)
        output_shape = []
        shape_a = self.get_func_input_shape(func.input[0])
        shape_b = self.get_func_input_shape(func.input[1])
        row_a = shape_a[len(shape_a) - 2]
        col_b = shape_b[len(shape_b) - 1]
        output_shape.extend(shape_a[:-2])
        output_shape.append(row_a)
        output_shape.append(col_b)
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def ElementWiseScalar(self, func_name, func_list, n):
        func = self.generate_default_function(func_name, n)
        if func_name == "RDivScalar":
            func_param = func.r_div_scalar_param
            func_param.val = 1.0
        elif func_name == "MulScalar":
            func_param = func.mul_scalar_param
            func_param.val = -1.0  # Neg is achieved by multiplying -1
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

    def LogSoftmax(self, func_list, n):
        func = self.generate_default_function("Sub2", n)
        logger.warning(SOFTMAX_WARNING)
        axis = DEFAULT_SOFTMAX_AXIS
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "LogSoftmax axis must be a single integer")
                axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        # Apply Exp+Sum+Log to the input,
        # and subtract the result with the original input
        lsin = n.input[0]
        expout = lsin+"_exp"
        expf = generate_unary("Exp", n.name, lsin, expout,
                              self._graph.name, self._func_counter)
        self._shape_output[expout] = self.get_func_input_shape(lsin)
        func_list.append(expf)
        sumout = expout+"_sum"
        # We keep dimension so the reduced sum can be subtracted
        # with the original input
        sumf = generate_sum(n.name, expout, sumout,
                            axis, True, self._graph.name, self._func_counter)
        output_shape = self.get_func_input_shape(expout)
        output_shape[axis] = 1
        self._shape_output[sumout] = output_shape
        func_list.append(sumf)
        logout = sumout+"_log"
        log = generate_unary("Log", n.name, sumout, logout,
                             self._graph.name, self._func_counter)
        self._shape_output[logout] = self.get_func_input_shape(sumout)
        func_list.append(log)
        # Rewire Sub2's input to the original input and
        # Exp+Sum+Log
        del func.input[:]
        func.input.extend([n.input[0], logout])
        input0_shape = self.get_func_input_shape(n.input[0])
        input1_shape = self.get_func_input_shape(logout)
        output_shape = []
        for i in range(len(input0_shape)):
            output_shape.append(max(input0_shape[i], input1_shape[i]))
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Softplus(self, func_list, n):
        # Convert to Exp+AddScalar+Log
        func = self.generate_default_function("Log", n)
        spin = n.input[0]
        expout = spin+"_exp"
        expf = generate_unary("Exp", n.name, spin,
                              expout, self._graph.name, self._func_counter)
        self._shape_output[expout] = self.get_func_input_shape(spin)
        func_list.append(expf)
        asout = expout+"_adds"
        asf = generate_add_scalar(n.name, expout, asout, 1.0,
                                  self._graph.name, self._func_counter)
        self._shape_output[asout] = self._shape_output[expout]
        func_list.append(asf)
        # rewire Log input to AddScalar output
        del func.input[:]
        func.input.extend([asout])
        self._shape_output[func.output[0]] = self._shape_output[asout]
        func_list.append(func)

    def Softsign(self, func_list, n):
        # Convert to Abs+AddScalar+Div2
        func = self.generate_default_function("Div2", n)
        ssin = n.input[0]
        expout = ssin+"_abs"
        expf = generate_unary("Abs", n.name, ssin,
                              expout, self._graph.name, self._func_counter)
        self._shape_output[expout] = self.get_func_input_shape(ssin)
        func_list.append(expf)
        asout = expout+"_adds"
        asf = generate_add_scalar(n.name, expout, asout, 1.0,
                                  self._graph.name, self._func_counter)
        self._shape_output[asout] = self._shape_output[expout]
        func_list.append(asf)
        # rewire Div2 input to original input and AddScalar output
        del func.input[:]
        func.input.extend([ssin, asout])
        input0_shape = self.get_func_input_shape(ssin)
        input1_shape = self.get_func_input_shape(asout)
        output_shape = []
        for i in range(len(input0_shape)):
            output_shape.append(max(input0_shape[i], input1_shape[i]))
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Clip(self, func_list, n):
        func = self.generate_default_function("Clip", n)
        maxval = None
        minval = None
        for attr in n.attribute:
            if attr.name == "max":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("max must be a single float")
                maxval = attr.f
            elif attr.name == "min":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("min must be a single float")
                minval = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        if maxval is None and minval is None:
            # No clipping. Convert to an identity
            func.type = "Identity"
            func_list.append(func)
        elif maxval is None and isinstance(minval, float):
            # Only min value is specified, so we convert to MaxScalar
            func.type = "MaximumScalar"
            msp = func.maximum_scalar_param
            msp.val = minval
            func_list.append(func)
        elif isinstance(maxval, float) and minval is None:
            # Only max value is specified, so we use MinScalar
            func.type = "MinimumScalar"
            msp = func.minimum_scalar_param
            msp.val = maxval
            func_list.append(func)
        else:  # both min and max is specified
            # Add MinimumScalar and rewire with MaximumScalar
            minin = n.input[0]
            minout = minin+"_min"
            minf = generate_minimum_scalar(n.name, minin, minout,
                                           maxval, self._graph.name, self._func_counter)
            self._shape_output[minout] = self.get_func_input_shape(minin)
            func_list.append(minf)
            func.type = "MaximumScalar"
            del func.input[:]
            func.input.extend([minout])
            msp = func.maximum_scalar_param
            msp.val = minval
            func_list.append(func)
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])

    def Unsqueeze(self, func_list, n):
        func = self.generate_default_function("Reshape", n)
        rp = func.reshape_param
        output_shape = self.get_func_input_shape(func.input[0])
        for attr in n.attribute:
            if attr.name == "axes":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for shape in {} op_type".format(n.op_type))
                for index in attr.ints:
                    output_shape.insert(index, 1)
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        rp.shape.dim.extend(output_shape)
        func_list.append(func)
        self._shape_output[func.output[0]] = output_shape

    def convert_to_functions(self, n):
        ft = self._onnx_optype_to_nnabla_function_type.get(n.op_type)
        if ft is None:
            raise ValueError(
                "op_type {} is currently not supported for NNP conversion".format(n.op_type))
        func_list = []
        if callable(ft):
            ft(func_list, n)
        return func_list

    def onnx_graph_to_nnp_protobuf(self, pb):
        network = pb.network.add()
        network.name = self._graph.name

        # convert nodes
        for n in self._graph.node:
            self.check_domain(n.domain)
            fl = self.convert_to_functions(n)

            network.function.extend(fl)

        # Gather all unique names for input and output
        for f in network.function:
            for i in f.input:
                self._all_vars[i] = None
            for o in f.output:
                self._all_vars[o] = None

        # convert parameters
        for init in self._graph.initializer:
            if init.name in self._merged_inputs:
                # Ignore any initializer that is already merged
                # to a function node
                continue
            add_tensor_as_parameter(pb, init)
            # Keep the list of all initializer names
            self._param_vars[init.name] = None
        # We need to distinguish constant parameters (which become 'Parameter' in NNabla)
        # from input/output variables (which become 'Buffer' in NNabla).
        # Constant parameters appear in the initializer list so we keep
        # all names of variables from the initializer and compare them with
        # the names we gathered in all_vars.
        # The names that only appear in all_vars are the input/output variables.

        # convert Input/Output ValueInfoProto
        # to Variable
        in_list = []  # list of input variables
        out_list = []  # list of output variables
        for i in self._graph.input:
            if i.name in self._merged_inputs:
                # Ignore any input that is already merged
                # to a function node
                continue
            if i.name in self._param_vars:
                # This input is a parameter
                v = add_value_info_as_parameter(network, i)
                self._param_list.append(v)
            else:
                # This input is a buffer
                v = add_value_info_as_buffer(network, i)
                in_list.append(v)
            if i.name in self._all_vars:
                del self._all_vars[i.name]
            else:
                # We come here when a buffer (usually a parameter) is included in
                # graph.input and graph.initializer,
                # but was not actually used as input for any node.
                # No one is using this buffer so we show a warning and ignore it.
                logger.warning(
                    "Input buffer {} is not used as input for any node.".format(i.name))
        for o in self._graph.output:
            if o.name in self._removed_outputs:
                # This output buffer was removed so we are not going to
                # use it
                continue
            v = add_value_info_as_buffer(network, o)
            out_list.append(v)
            del self._all_vars[v.name]

        for varg in self._all_vars:
            # We add all remaining variables as intermediate buffer,
            # except for the ones that was converted to a parameter.
            # A conversion of a buffer to a parameter may occur when functions
            # such as Constant is given.
            if varg in self._param_vars:
                continue
            v = network.variable.add()
            v.type = "Buffer"
            v.name = varg
            if varg in self._shape_output:
                v.shape.dim.extend(self._shape_output[varg])

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
        for pv in self._param_list:
            p = exe.parameter_variable.add()
            p.variable_name = pv.name
        convert_parameter_shape(pb)

    def onnx_model_to_nnp_protobuf(self):
        pb = nnabla_pb2.NNablaProtoBuf()
        self._pb = pb
        if self._ir_version > ONNX_IR_VERSION:
            raise ValueError("ONNX IR version newer than {} is currently not supported: {}".format(
                ONNX_IR_VERSION, self._ir_version))
        for opset in self._opset_import:
            if opset.domain == "":
                # ONNX opset.
                if opset.version <= 6:
                    self._onnx_optype_to_nnabla_function_type = self.opver_impl_map.get(
                        "6")
                else:
                    self._onnx_optype_to_nnabla_function_type = self.opver_impl_map.get(
                        str(opset.version))
                if not self._onnx_optype_to_nnabla_function_type:
                    raise ValueError("ONNX opset version is currently not supported: {}".format(
                        opset.version))
            else:
                raise ValueError(
                    "Unsupported opset from domain {}".format(opset.domain))

        # convert onnx model to nnabla protobuf
        # logger.log(99, "Converting ONNX made by {}.".format(model.producer_name))

        # convert graph
        self.onnx_graph_to_nnp_protobuf(pb)

        class nnp:
            pass
        nnp.protobuf = pb
        nnp.other_files = []
        return nnp

    def import_from_onnx_model(self, onnx_model):
        self._ir_version = onnx_model.ir_version
        self._graph = onnx_model.graph
        self._opset_import = onnx_model.opset_import

    def execute(self):
        if self._file_path != '':
            self.get_onnx_graph_info()
        return self.onnx_model_to_nnp_protobuf()
