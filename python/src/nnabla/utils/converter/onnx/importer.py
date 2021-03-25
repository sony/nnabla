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


# ONNX does not have the concept of executors.
# We will add a single executor to NNP when converted from ONNX,
# and set a default name to it.
DEFAULT_EXECUTOR_NAME = "exec_0"

fork_name_number = 0


def add_value_info_as_variable(network, info):
    if not info.type.HasField("tensor_type"):  # accepting only tensor
        raise ValueError("Only TensorProto is allowed as ValueInfoProto's type for info.name (Got {})"
                         .format(info.name, info.type))
    t = info.type.tensor_type
    v = network.variable.add()
    v.name = info.name
    v.shape.dim.extend(
        [x.dim_value if not x.dim_param else -1 for x in t.shape.dim])
    return v


def add_value_info_as_buffer(network, info):
    v = add_value_info_as_variable(network, info)
    v.type = "Buffer"
    return v


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


def fork_name(name):
    global fork_name_number
    fork_name_number += 1
    ret = name + '_{:04}'.format(fork_name_number)
    return ret


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
    """Generate a BroadcastTo operator to brodcastto specified buffer"""
    bt = nnabla_pb2.Function()
    bt.type = "BroadcastTo"
    set_function_name(bt, node_name, base_name, func_counter)
    bt.input.extend([x, y])
    bt.output.extend([out_name])
    btp = bt.broadcast_to_param
    btp.axis = axis
    return bt


def generate_broadcast(node_name, in_name, out_name, shape, base_name, func_counter):
    """Generate a Broadcast operator to brodcast specified buffer"""
    bt = nnabla_pb2.Function()
    bt.type = "Broadcast"
    set_function_name(bt, node_name, base_name, func_counter)
    bt.input.extend([in_name])
    bt.output.extend([out_name])
    btp = bt.broadcast_param
    btp.shape.dim.extend(shape)
    return bt


def generate_split(node_name, in_name, out_name, axis, base_name, func_counter):
    """Generate a Split operator to split specified buffer"""
    sp = nnabla_pb2.Function()
    sp.type = "Split"
    set_function_name(sp, node_name, base_name, func_counter)
    sp.input.extend([in_name])
    sp.output.extend(out_name)
    spp = sp.split_param
    spp.axis = axis
    return sp


def generate_stack(node_name, in_name, out_name, axis, base_name, func_counter):
    """Generate a Stack operator to stack specified buffer"""
    sp = nnabla_pb2.Function()
    sp.type = "Stack"
    set_function_name(sp, node_name, base_name, func_counter)
    sp.input.extend(in_name)
    sp.output.extend([out_name])
    spp = sp.stack_param
    spp.axis = axis
    return sp


def generate_unary(func_name, node_name, x,
                   out_name, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = func_name
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    return func


def generate_arithmetic(func_name, node_name, inputs,
                        out_name, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = func_name
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend(inputs)
    func.output.extend([out_name])
    return func


def generate_reduction(func_name, node_name, x, out_name,
                       axes, keepdims, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = func_name
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    if func_name == "Max":
        sp = func.max_param
    elif func_name == "Sum":
        sp = func.sum_param
    sp.axes.extend(axes)
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


def add_tensor_as_parameter(pb, tensor):
    """Add given tensor as a parameter"""
    p = pb.parameter.add()
    p.variable_name = tensor.name
    shape = tensor.dims
    if not shape:
        shape = [1]
    p.shape.dim.extend(shape)
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
    elif tensor.data_type == TensorProto.INT8:
        if tensor.raw_data:
            p.data.extend(np.fromstring(tensor.raw_data, dtype=np.int8))
        elif len(tensor.int32_data) > 0:
            p.data.extend(tensor.int32_data)
        else:
            raise ValueError("int8 data not found for {}".format(tensor.name))
    elif tensor.data_type == TensorProto.UINT8:
        if tensor.raw_data:
            p.data.extend(np.fromstring(tensor.raw_data, dtype=np.uint8))
        elif len(tensor.int32_data) > 0:
            p.data.extend(tensor.int32_data)
        else:
            raise ValueError("uint8 data not found for {}".format(tensor.name))
    else:
        raise ValueError("Unsupported tensor data type for {}: {}"
                         .format(tensor.name, tensor.data_type))
    p.need_grad = False

    # Add tensor as variable
    v = pb.network[0].variable.add()
    v.name = tensor.name
    v.shape.dim.extend(shape)
    v.type = "Parameter"
    v.initializer.type = "Constant"
    v.initializer.multiplier = 1.0


def create_parameter_variable(pb, name, shape, val):
    p = pb.parameter.add()
    p.variable_name = name
    p.shape.dim.extend(shape)
    p.data.extend(val)

    v = pb.network[0].variable.add()
    v.name = name
    v.shape.dim.extend(shape)
    v.type = "Parameter"
    v.initializer.type = "Constant"
    v.initializer.multiplier = 1.0

    return v


class OnnxImporter:
    def __init__(self, file_path=''):
        self._file_path = file_path

        # We use an OrderedDict and not a set
        # to preserve order
        self._param_vars = OrderedDict()  # Dictionary for input parameters.
        self._all_vars = OrderedDict()  # Dictionary for all variables
        self._merged_inputs = []  # list of input buffers that was merged to a function
        self._removed_outputs = []  # list of output buffers that was removed
        self._func_counter = {}  # a counter for all functions
        self._shape_output = {}  # The shape of the output that all functions
        self._cast_node = {}  # Dict holding Cast node info, format: {output:input}
        # Convert the shape of some parameters, format: {name: new_shape}
        self._parameter_shape = {}

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
            "Pad": partial(self.Pad, '6'),
            "Relu": partial(self.GeneralOperator, 'ReLU'),
            "PRelu": self.PRelu,
            "Concat": self.Concatenate,
            "Conv": self.Convolution,
            "GlobalAveragePool": partial(self.BasePooling, 'GlobalAveragePooling'),
            "MaxPool": partial(self.BasePooling, 'MaxPooling'),
            "AveragePool": partial(self.BasePooling, 'AveragePooling'),
            "Sum": partial(self.GeneralOperator, 'AddN'),
            "Gemm": self.Gemm,
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
            "Min": partial(self.GeneralOperator, 'Minimum2'),
            "Max": partial(self.GeneralOperator, 'Maximum2'),
            "Reciprocal": partial(self.ElementWiseScalar, 'RDivScalar'),
            "Neg": partial(self.ElementWiseScalar, 'MulScalar'),
            "LogSoftmax": self.LogSoftmax,
            "Softplus": partial(self.GeneralOperator, 'SoftPlus'),
            "Softsign": partial(self.GeneralOperator, 'SoftSign'),
            "LRN": self.LRN,
            "Clip": partial(self.Clip, 6),
            # Constant does not get converted to a function
            # but we list it here so we can accept it
            "Constant": self.Constant,
            "Unsqueeze": self.Unsqueeze,
            "Sqrt": self.Sqrt,
            "Ceil": partial(self.GeneralOperator, 'Ceil'),
            "Floor": partial(self.GeneralOperator, 'Floor'),
            "Tile": self.Tile,
            "Flatten": self.Flatten,
            "Squeeze": self.Squeeze,
            "Slice": partial(self.Slice, '6'),
            # Currently, caffe2 does not support this function.
            "DepthToSpace": self.DepthToSpace,
            "SpaceToDepth": self.SpaceToDepth,
            "ArgMax": partial(self.ElementIndices, "Max"),
            "ArgMin": partial(self.ElementIndices, "Min"),
            "Split": self.Split,
            "Upsample": self.Upsample_6,
            "Mean": self.Mean,
            "ConvTranspose": self.ConvTranspose,
            "ConstantOfShape": self.ConstantOfShape,
            "HardSigmoid": self.HardSigmoid,
            "Hardmax": self.Hardmax,
            "InstanceNormalization": self.InstanceNormalization,
            "ReduceSumSquare": self.ReduceSumSquare,
            # Currently, Cast does not get converted to a function
            # but we list it here so we can accept it
            "Cast": self.Cast,
            "Gather": self.Gather,
        }

        # opset_7 table
        self.table_op_set_7 = {
            "Dropout": partial(self.Dropout, 7),
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
            "Acos": partial(self.GeneralOperator, 'ACos'),
            "Asin": partial(self.GeneralOperator, 'ASin'),
            "Atan": partial(self.GeneralOperator, 'ATan'),
            "Cos": partial(self.GeneralOperator, 'Cos'),
            "Sin": partial(self.GeneralOperator, 'Sin'),
            "Tan": partial(self.GeneralOperator, 'Tan'),
            "Upsample": self.Upsample_7,
        }
        self.table_op_set_7 = dict(self.table_op_set_6, **self.table_op_set_7)

        # opset_9 table
        self.table_op_set_9 = {
            "Min": partial(self.BroadcastOperator_9, 'Minimum2'),
            "Max": partial(self.BroadcastOperator_9, 'Maximum2'),
            # Currently, caffe2 does not support this function.
            "Acosh": partial(self.GeneralOperator, 'ACosh'),
            # Currently, caffe2 does not support this function.
            "Asinh": partial(self.GeneralOperator, 'ASinh'),
            # Currently, caffe2 does not support this function.
            "Atanh": partial(self.GeneralOperator, 'ATanh'),
            "Cosh": partial(self.GeneralOperator, 'Cosh'),
            "Sinh": partial(self.GeneralOperator, 'Sinh'),
            "IsNaN": partial(self.GeneralOperator, 'IsNaN'),
            "Sign": partial(self.GeneralOperator, 'Sign'),
            "Upsample": self.Upsample_9,
            "Expand": self.Expand,
            "Where": partial(self.GeneralOperator, 'Where'),
        }
        self.table_op_set_9 = dict(self.table_op_set_7, **self.table_op_set_9)

        # opset_10 table
        self.table_op_set_10 = {
            "ThresholdedRelu": self.ThresholdedRelu,
            "IsInf": partial(self.GeneralOperator, 'IsInf'),
            "Slice": partial(self.Slice, '10'),
            "QuantizeLinear": self.QuantizeLinear,
            "DequantizeLinear": self.DequantizeLinear,
        }
        self.table_op_set_10 = dict(
            self.table_op_set_9, **self.table_op_set_10)

        # opset_11 table
        self.table_op_set_11 = {
            "Clip": partial(self.Clip, 11),
            "Round": partial(self.GeneralOperator, 'Round'),
            "Pad": partial(self.Pad, '11'),
        }
        self.table_op_set_11 = dict(
            self.table_op_set_10, **self.table_op_set_11)

        # Currently, we only planed to support opset 6 and opset 11.
        # More planes will be added later to support more opset versions.
        self.opver_impl_map = {
            "6": self.table_op_set_6,
            "7": self.table_op_set_7,
            "9": self.table_op_set_9,
            "10": self.table_op_set_10,
            "11": self.table_op_set_11,
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
        if input_name in self._cast_node:
            input_name = self._cast_node[input_name]
        if input_name in self._shape_output:
            input_shape.extend(self._shape_output[input_name])
        else:
            for i in self._graph.input:
                if i.name == input_name:
                    t = i.type.tensor_type
                    if len(t.shape.dim):
                        input_shape = [
                            x.dim_value if not x.dim_param else 1 for x in t.shape.dim]
                    else:
                        input_shape = [1]
                    return input_shape
            for i in self._graph.initializer:
                if i.name == input_name:
                    if len(i.dims):
                        input_shape = list(i.dims)
                    else:
                        input_shape = [1]
                    return input_shape
        if not input_shape:
            raise ValueError(
                "The shape of {} was not found".format(input_name))
        return input_shape

    def get_input_raw_data(self, input_name, data_type):
        data = []

        # Try to find data in constant node
        for op in self._graph.node:
            if op.output[0] == input_name and op.op_type == "Constant":
                for attr in op.attribute:
                    if attr.name == "value":
                        if attr.t.data_type == TensorProto.INT64:
                            if attr.t.raw_data:
                                data.extend(np.fromstring(
                                    attr.t.raw_data, dtype=np.int64))
                            elif attr.t.int64_data:
                                data.extend(attr.t.int64_data)
                            break
                        elif attr.t.data_type == TensorProto.FLOAT:
                            if attr.t.raw_data:
                                data.extend(np.fromstring(
                                    attr.t.raw_data, dtype=np.float32))
                            elif attr.t.float_data:
                                data.extend(attr.t.float_data)
                            break
                        elif attr.t.data_type == TensorProto.BOOL:
                            if attr.t.raw_data:
                                data.extend(np.fromstring(
                                    attr.t.raw_data, dtype=np.bool))
                            break

        # Try to find data in the initializer.
        for init in self._graph.initializer:
            if init.name == input_name:
                if data_type == TensorProto.INT64:
                    if init.raw_data:
                        data.extend(np.fromstring(
                            init.raw_data, dtype=np.int64))
                    elif init.int64_data:
                        data.extend(init.int64_data)
                    break
                elif data_type == TensorProto.FLOAT:
                    if init.raw_data:
                        data.extend(np.fromstring(
                            init.raw_data, dtype=np.float32))
                    elif init.float_data:
                        data.extend(init.float_data)
                    break
                elif data_type == TensorProto.BOOL:
                    if init.raw_data:
                        data.extend(np.fromstring(
                            init.raw_data, dtype=np.bool))
                    break

        if not data:
            raise ValueError("Not found {}".format(input_name))

        self._merged_inputs.append(input_name)
        return data

    def generate_default_function(self, func_name, n):
        func = nnabla_pb2.Function()
        func.type = func_name
        set_function_name(func, n.name, self._graph.name, self._func_counter)
        func.input.extend(n.input)
        func.output.extend(n.output)
        return func

    def generate_expand_batchmatmul(self, node_name, inputs, output, transpose):
        def batchmatmul(in_names, out_name):
            bm = nnabla_pb2.Function()
            bm.type = "BatchMatmul"
            set_function_name(bm, node_name, self._graph.name,
                              self._func_counter)
            bm.input.extend(in_names)
            bm.output.extend(out_name)
            bmp = bm.batch_matmul_param
            bmp.transpose_a = transpose[0]
            bmp.transpose_b = transpose[1]
            return bm

        f_list = []
        transA_shape = self.get_func_input_shape(inputs[0])
        transB_shape = self.get_func_input_shape(inputs[1])
        output_shape = []
        output_shape.append(
            transA_shape[-1] if transpose[0] else transA_shape[-2])
        output_shape.append(
            transB_shape[-2] if transpose[1] else transB_shape[-1])
        assert len(transA_shape) == len(
            transB_shape), "ndim of inputs[0] must be ndim of inputs[1]."
        if len(transA_shape) < 3:
            # Expand input[0]
            transA_rp_out = fork_name(inputs[0]) + "_reshape"
            rp = generate_reshape(node_name, inputs[0], transA_rp_out,
                                  [1] + transA_shape, self._graph.name, self._func_counter)
            self._shape_output[transA_rp_out] = [1] + transA_shape
            f_list.append(rp)

            # Expand input[1]
            transB_rp_out = fork_name(inputs[1]) + "_reshape"
            rp = generate_reshape(node_name, inputs[1], transB_rp_out,
                                  [1] + transB_shape, self._graph.name, self._func_counter)
            self._shape_output[transB_rp_out] = [1] + transB_shape
            f_list.append(rp)

            # BatchMatmul
            matmul_out = fork_name(output[0]) + "_batchmatmul"
            bm = batchmatmul([transA_rp_out, transB_rp_out], [matmul_out])
            self._shape_output[matmul_out] = [1] + output_shape
            f_list.append(bm)

            # Reshape
            rp = generate_reshape(node_name, matmul_out, output[0],
                                  output_shape, self._graph.name, self._func_counter)
            f_list.append(rp)
            self._shape_output[output[0]] = output_shape
        else:
            bm = batchmatmul(inputs, output)
            self._shape_output[output[0]] = output_shape
            f_list.append(bm)

        return f_list

    def Convolution(self, func_list, n):
        func = self.generate_default_function("Convolution", n)
        cp = func.convolution_param
        input_shape = self.get_func_input_shape(func.input[0])
        weight_shape = self.get_func_input_shape(func.input[1])
        # We shouldn't need these default settings
        # since NNabla will set these for us
        cp.base_axis = 1
        cp.group = 1
        dims = len(input_shape) - 2
        pads = [0] * dims * 2
        strides = [1] * dims
        dilations = [1] * dims
        auto_pad = "NOTSET"
        for attr in n.attribute:
            if attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for pads in Conv op_type")
                pads.clear()
                pads.extend(attr.ints)
            elif attr.name == "strides":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for strides in Conv op_type")
                strides.clear()
                strides.extend(attr.ints)
            elif attr.name == "dilations":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for dilations in Conv op_type")
                dilations.clear()
                dilations.extend(attr.ints)
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
            elif attr.name == "auto_pad":
                if attr.type != AttributeProto.STRING:
                    raise ValueError("Only STRING is supported for auto_pad in {} op_type"
                                     .format(n.op_type))
                auto_pad = attr.s.decode("utf-8")
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        # NNabla requires for the dimensions of strides, pads, dilations to match.
        # We align the dimensions for all three attributes to the shortest one
        cp.stride.dim.extend(strides[:])
        cp.dilation.dim.extend(dilations[:])
        if auto_pad != 'NOTSET':
            o = input_shape[2:]
            k = weight_shape[2:]
            pads_value = [0] * dims
            if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                for i in range(dims):
                    pads_value[i] = (o[i] - 1) * strides[i] + \
                                     k[i] - input_shape[2 + i]
            elif auto_pad == 'VALID':
                pass
            for i in range(dims):
                if auto_pad == 'SAME_LOWER':
                    pads[i + dims] = pads_value[i] // 2
                    pads[i] = pads_value[i] - pads[i + dims]
                elif auto_pad == 'SAME_UPPER':
                    pads[i] = pads_value[i] // 2
                    pads[i + dims] = pads_value[i] - pads[i + dims]

        padval = []
        asymmetry = check_padding(pads, dims, padval)
        if asymmetry:
            # Add a separate padding function for
            # asymmetry padding
            input = n.input[0]
            padded = fork_name(input)+"_pad"
            pad_width = rearrange_pads(pads)
            padf = generate_pad(n.name, input, padded,
                                "constant", pad_width, 0,
                                self._graph.name, self._func_counter)
            for i in range(dims):
                input_shape[2 + i] += pad_width[2 * i]
                input_shape[2 + i] += pad_width[2 * i + 1]
            self._shape_output[padded] = input_shape
            func_list.append(padf)
            # Rewire input to the padded version
            func.input[0] = padded
        cp.pad.dim.extend(padval)

        output_shape = input_shape[:1]
        output_shape.append(weight_shape[0])
        for index in range(dims):
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
        input_shape = self.get_func_input_shape(n.input[0])
        scale_shape = self.get_func_input_shape(n.input[1])
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
        new_const_inp_shape = [1] * len(input_shape)
        new_const_inp_shape[1] = scale_shape[0]
        for inp in n.input[1:5]:
            self._parameter_shape[inp] = new_const_inp_shape
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

    def PRelu(self, func_list, n):
        slope_shape = self.get_func_input_shape(n.input[1])
        if len(slope_shape) != 1:
            raise ValueError(
                "Only one dimensional is currently supported for PRelu input tensor slope")
        func = self.generate_default_function("PReLU", n)
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
        input_shape = self.get_func_input_shape(func.input[0])
        rp = func.reshape_param
        new_shape = []
        shape_found = False
        for attr in n.attribute:
            if attr.name == "shape":
                # Shape comes as attribute for Reshape-1
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for shape in {} op_type".format(n.op_type))
                new_shape.extend(attr.ints)
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
            raw_data = self.get_input_raw_data(shape_input, TensorProto.INT64)
            new_shape.extend(raw_data)
            shape_found = True
            # stored the merged input so we can ignore it later
            del func.input[1]
        if not shape_found:
            raise ValueError(
                "Shape information was not found in {} op_type".format(n.op_type))

        if -1 in new_shape:
            shape_infer_index = -1
            reset_size = 1
            for i, s in enumerate(new_shape):
                if s < 0:
                    if shape_infer_index >= 0:
                        raise ValueError(
                            'Reshape: shape has multiple negative number.')
                    shape_infer_index = i
                else:
                    reset_size *= s
            new_shape[shape_infer_index] = int(
                np.prod(input_shape) / reset_size)
        rp.shape.dim.extend(new_shape)
        self._shape_output[func.output[0]] = new_shape
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
        def get_pad_shape(auto_pad,  # type: Text
                          input_spatial_shape,  # type: Sequence[int]
                          kernel_spatial_shape,  # type: Sequence[int]
                          strides_spatial,  # type: Sequence[int]
                          output_spatial_shape  # type: Sequence[int]
                          ):  # type: (...) -> Sequence[int]
            pad_shape = [0] * len(input_spatial_shape)
            if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                for i in range(len(input_spatial_shape)):
                    pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                        kernel_spatial_shape[i] - input_spatial_shape[i]
            elif auto_pad == 'VALID':
                pass
            return pad_shape

        def get_output_shape(auto_pad,  # type: Text
                             input_spatial_shape,  # type: Sequence[int]
                             kernel_spatial_shape,  # type: Sequence[int]
                             strides_spatial  # type: Sequence[int]
                             ):  # type: (...) -> Sequence[int]
            out_shape = [0] * len(input_spatial_shape)
            if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                for i in range(len(input_spatial_shape)):
                    out_shape[i] = int(
                        np.ceil(
                            float(
                                input_spatial_shape[i])
                            / float(
                                strides_spatial[i])))
            elif auto_pad == 'VALID':
                for i in range(len(input_spatial_shape)):
                    out_shape[i] = int(np.ceil(float(
                        input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
            return out_shape

        def cal_pads(auto_pad, pad_shape):
            spatial_size = len(pad_shape)
            pads = [0] * spatial_size * 2
            for i in range(spatial_size):
                if auto_pad == "SAME_LOWER":
                    pads[i + spatial_size] = pad_shape[i] // 2
                    pads[i] = pad_shape[i] - pads[i + spatial_size]
                elif auto_pad == "SAME_UPPER":
                    pads[i] = pad_shape[i] // 2
                    pads[i + spatial_size] = pad_shape[i] - pads[i]
            return pads

        def set_kernel_parameter_and_get_pads(node,
                                              kp,
                                              input_spatial_shape
                                              ):
            strides = []
            pads = []
            kernel = []
            auto_pad = "NOTSET"

            for attr in node.attribute:
                if attr.name == "strides":
                    if attr.type != AttributeProto.INTS:
                        raise ValueError("Only INTS are supported for strides in {}"
                                         .format(node.op_type))
                    strides.extend(attr.ints)
                elif attr.name == "pads":
                    if attr.type != AttributeProto.INTS:
                        raise ValueError("Only INTS are supported for pads in {}"
                                         .format(node.op_type))
                    pads.extend(attr.ints)
                elif attr.name == "kernel_shape":
                    if attr.type != AttributeProto.INTS:
                        raise ValueError("Only INTS are supported for kernel_shape in {}"
                                         .format(node.op_type))
                    kernel.extend(attr.ints)
                elif attr.name == "count_include_pad":
                    if attr.type != AttributeProto.INT:
                        raise ValueError("Only INT is supported for count_include_pad in {} op_type"
                                         .format(node.op_type))
                    kp.including_pad = bool(attr.i)
                elif attr.name == "auto_pad":
                    if attr.type != AttributeProto.STRING:
                        raise ValueError("Only STRING is supported for auto_pad in {} op_type"
                                         .format(node.op_type))
                    auto_pad = attr.s.decode("utf-8")
                else:
                    raise ValueError("Unsupported attribute {} was specified at {}"
                                     .format(attr.name, node.op_type))

            if kernel:
                kp.kernel.dim.extend(kernel[:])
            else:
                raise ValueError("kernel_shape is required")
            if strides:
                kp.stride.dim.extend(strides[:])
            else:
                kp.stride.dim.extend([1]*len(kp.kernel.dim))

            if not pads and auto_pad != "NOTSET":
                output_spatial_shape = get_output_shape(
                    auto_pad, input_spatial_shape, kp.kernel.dim, kp.stride.dim)
                pad_shape = get_pad_shape(
                    auto_pad, input_spatial_shape, kp.kernel.dim, kp.stride.dim, output_spatial_shape)
                pads = cal_pads(auto_pad, pad_shape)

            return pads

        input_shape = self.get_func_input_shape(n.input[0])
        pad_mode = "constant"
        value = 0.0

        if func_name == 'AveragePooling':
            func = self.generate_default_function("AveragePooling", n)
            kp = func.average_pooling_param
            pad_mode = "replicate"
        elif func_name == 'MaxPooling':
            func = self.generate_default_function("MaxPooling", n)
            kp = func.max_pooling_param
            value = -np.inf
        elif func_name == 'GlobalAveragePooling':
            func = self.generate_default_function("GlobalAveragePooling", n)
            input_shape = self.get_func_input_shape(func.input[0])
            output_shape = []
            output_shape.extend(input_shape[:2])
            output_shape.extend([1, 1])
            self._shape_output[func.output[0]] = output_shape
            func_list.append(func)
            return

        padval = []
        pads = set_kernel_parameter_and_get_pads(n, kp, input_shape[2:])
        kp.ignore_border = True
        dim = len(kp.kernel.dim)
        if func_name == "AveragePooling":
            if kp.including_pad:
                pad_mode = "constant"
        if pads:
            asymmetry = check_padding(pads, dim, padval)
            if asymmetry:
                pad_width = rearrange_pads(pads)
                input = n.input[0]
                pad_out = fork_name(input) + "_pad"
                padf = generate_pad(n.name, input, pad_out,
                                    pad_mode, pad_width, value,
                                    self._graph.name, self._func_counter)
                for i in range(dim):
                    input_shape[2 + i] += pad_width[2 * i]
                    input_shape[2 + i] += pad_width[2 * i + 1]
                self._shape_output[pad_out] = input_shape
                func_list.append(padf)
                del func.input[:]
                func.input.extend([pad_out])
            kp.pad.dim.extend(padval)
        else:
            kp.pad.dim.extend([0]*dim)
        output_shape = input_shape[:]
        for i in range(dim):
            output_shape[i + 2] = (input_shape[i + 2] + 2 *
                                   kp.pad.dim[i] - kp.kernel.dim[i]) // kp.stride.dim[i] + 1
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Concatenate(self, func_list, n):
        # Concat axis was not required for Concat-1 (it is required from Concat-4),
        # so the default axis depended on which backend we use.
        # Since we are comparing with caffe2, we are
        # defaulting to the channel axis if the axis is not specified.
        # https://github.com/onnx/onnx/issues/374
        func = self.generate_default_function("Concatenate", n)
        input_shape = self.get_func_input_shape(func.input[0])
        func.concatenate_param.axis = 1
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Axis type must be a single integer")
                # The axis was specified so we use it
                if attr.i < 0:
                    func.concatenate_param.axis = len(input_shape) + attr.i
                else:
                    func.concatenate_param.axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
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
        bout = fork_name(bin)+broadcasted_postfix
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
        # NNabla can only process two inputs for max/min.
        # Check if this is fulfilled.
        if len(n.input) != 2:
            raise ValueError(
                "NNabla can only process Min/Max of two tensors")
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
                    out = fork_name(n.input[0])+"_shape"
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
                    out = fork_name(n.input[1])+"_shape"
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

    def Gemm(self, func_list, n):
        alpha = 1.0
        beta = 1.0
        transpose_a = 0
        transpose_b = 0
        inputs = n.input[:]
        transA_shape = self.get_func_input_shape(inputs[0])
        transB_shape = self.get_func_input_shape(inputs[1])
        shape = [transA_shape[0], transB_shape[1]]

        # Switch H and W for transpose
        # We assume the buffer is two dimensional.
        for attr in n.attribute:
            if attr.name == "transA":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for transA in {} op_type".format(n.op_type))
                if attr.i:
                    shape[0] = transA_shape[1]
                    transpose_a = attr.i
            elif attr.name == "transB":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for transB in {} op_type".format(n.op_type))
                if attr.i:
                    shape[1] = transB_shape[0]
                    transpose_b = attr.i
            elif attr.name == "broadcast":
                pass
            elif attr.name == "alpha":
                alpha = attr.f
            elif attr.name == "beta":
                beta = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        if alpha != 1.0:
            # MulScalar
            muls_out = fork_name(inputs[0])+"_muls"
            muls = generate_mul_scalar(n.name, inputs[0], muls_out,
                                       alpha, self._graph.name, self._func_counter)
            self._shape_output[muls_out] = transA_shape
            func_list.append(muls)
            inputs[0] = muls_out

        if len(inputs) < 3:
            expand_bm = self.generate_expand_batchmatmul(
                n.name, inputs, n.output, [transpose_a, transpose_b])
            func_list.extend(expand_bm)
        else:
            bias_shape = self.get_func_input_shape(inputs[2])
            if beta != 1.0:
                # MulScalar
                muls_out = fork_name(inputs[2])+"_muls"
                muls = generate_mul_scalar(n.name, inputs[2], muls_out,
                                           beta, self._graph.name, self._func_counter)
                self._shape_output[muls_out] = bias_shape
                func_list.append(muls)
                inputs[2] = muls_out

            expand_bm_out = fork_name(n.output[0]) + "_bm"
            expand_bm = self.generate_expand_batchmatmul(
                n.name, inputs, [expand_bm_out], [transpose_a, transpose_b])
            func_list.extend(expand_bm)

            if len(bias_shape) != len(shape):
                rout = fork_name(inputs[2]) + "_reshape"
                bias_shape = [1] * (len(shape) - len(bias_shape)) + bias_shape
                rp = generate_reshape(n.name, inputs[2], rout, bias_shape,
                                      self._graph.name, self._func_counter)
                func_list.append(rp)
                self._shape_output[rout] = bias_shape
                inputs[2] = rout

            add2_func = self.generate_default_function("Add2", n)
            del add2_func.input[:]
            add2_func.input.extend([expand_bm_out, inputs[2]])
            func_list.append(add2_func)
            self._shape_output[n.output[0]] = shape

    def Softmax(self, func_list, n):
        axis = 1
        input_shape = self.get_func_input_shape(n.input[0])
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Softmax axis must be a single integer")
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        # Reshape
        rout_x = fork_name(n.input[0])+"_reshape"
        _shape = [int(np.prod(input_shape[:axis])),
                  int(np.prod(input_shape[axis:]))]
        rp = generate_reshape(n.name, n.input[0], rout_x, _shape,
                              self._graph.name, self._func_counter)
        self._shape_output[rout_x] = _shape
        func_list.append(rp)

        # Max
        mout_x = fork_name(rout_x)+"_max"
        gr = generate_reduction("Max", n.name, rout_x, mout_x,
                                [1], True, self._graph.name, self._func_counter)
        self._shape_output[mout_x] = [_shape[0], 1]
        func_list.append(gr)

        # Sub2
        sout_x = fork_name(rout_x)+"_sub2"
        ga = generate_arithmetic("Sub2", n.name, [rout_x, mout_x], sout_x,
                                 self._graph.name, self._func_counter)
        self._shape_output[sout_x] = _shape
        func_list.append(ga)

        # Exp
        expout_x = fork_name(sout_x)+"_exp"
        ge = generate_unary("Exp", n.name, sout_x, expout_x,
                            self._graph.name, self._func_counter)
        self._shape_output[expout_x] = _shape
        func_list.append(ge)

        # Sum
        sumout_x = fork_name(expout_x)+"_sum"
        gr = generate_reduction("Sum", n.name, expout_x, sumout_x,
                                [1], True, self._graph.name, self._func_counter)
        self._shape_output[sumout_x] = [_shape[0], 1]
        func_list.append(gr)

        # Div2
        div2out_x = fork_name(n.output[0])+"_div2"
        ga = generate_arithmetic("Div2", n.name, [expout_x, sumout_x], div2out_x,
                                 self._graph.name, self._func_counter)
        self._shape_output[div2out_x] = _shape
        func_list.append(ga)

        # Reshape
        rp = generate_reshape(n.name, div2out_x, n.output[0], input_shape,
                              self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = input_shape
        func_list.append(rp)

    def Pad(self, opset, func_list, n):
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
        if opset == "11":
            del func.input[1:]
            pads = self.get_input_raw_data(n.input[1], TensorProto.INT64)
            try:
                value = self.get_input_raw_data(
                    n.input[2], TensorProto.FLOAT)[0]
            except:
                pass

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
        pow0_out = fork_name(pow0_in)+"_pow0"
        pow0 = generate_pow_scalar(n.name, pow0_in, pow0_out,
                                   2, self._graph.name, self._func_counter)
        self._shape_output[pow0_out] = self.get_func_input_shape(pow0_in)
        func_list.append(pow0)
        # Transpose the channel axis so we can sumpool along the channels
        # We are assuming 4D input
        trans0_out = fork_name(pow0_out)+"_trans0"
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
        sp_out = fork_name(trans0_out)+"_sp"
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
        trans1_out = fork_name(sp_out)+"_trans1"
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
        muls_out = fork_name(trans1_out)+"_muls"
        muls = generate_mul_scalar(n.name, trans1_out, muls_out,
                                   alpha/size, self._graph.name, self._func_counter)
        self._shape_output[muls_out] = self._shape_output[trans1_out]
        func_list.append(muls)
        # AddScalar
        adds_out = fork_name(muls_out)+"_adds"
        adds = generate_add_scalar(n.name, muls_out, adds_out,
                                   bias, self._graph.name, self._func_counter)
        self._shape_output[adds_out] = self._shape_output[muls_out]
        func_list.append(adds)
        # PowScalar
        pow1_out = fork_name(adds_out)+"_pow1"
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
            if len(func_param.axes) == 0:
                func_param.axes.extend(range(len(output_shape)))
            else:
                for i, axis in enumerate(func_param.axes):
                    if axis < 0:
                        func_param.axes[i] = len(output_shape) + axis
            if func_param.keep_dims:
                output_shape = [1 if i in func_param.axes else output_shape[i]
                                for i in range(len(output_shape))]
            else:
                output_shape = [output_shape[i] for i in range(
                    len(output_shape)) if i not in func_param.axes]
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
                if not t.dims:
                    t.dims.extend([1])
                # add tensor as parameter
                add_tensor_as_parameter(self._pb, t)
                self._param_vars[t.name] = None
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
        expand_bm = self.generate_expand_batchmatmul(
            n.name, n.input, n.output, [0, 0])
        func_list.extend(expand_bm)

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
        axis = 1
        input_shape = self.get_func_input_shape(n.input[0])
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "LogSoftmax axis must be a single integer")
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        # Reshape
        rout_x = fork_name(n.input[0])+"_reshape"
        _shape = [int(np.prod(input_shape[:axis])),
                  int(np.prod(input_shape[axis:]))]
        rp = generate_reshape(n.name, n.input[0], rout_x, _shape,
                              self._graph.name, self._func_counter)
        self._shape_output[rout_x] = _shape
        func_list.append(rp)

        # Max
        mout_x = fork_name(rout_x)+"_max"
        gr = generate_reduction("Max", n.name, rout_x, mout_x,
                                [1], True, self._graph.name, self._func_counter)
        self._shape_output[mout_x] = [_shape[0], 1]
        func_list.append(gr)

        # Sub2
        sout_x = fork_name(rout_x)+"_sub2"
        ga = generate_arithmetic("Sub2", n.name, [rout_x, mout_x], sout_x,
                                 self._graph.name, self._func_counter)
        self._shape_output[sout_x] = _shape
        func_list.append(ga)

        # Exp
        expout_x = fork_name(sout_x)+"_exp"
        ge = generate_unary("Exp", n.name, sout_x, expout_x,
                            self._graph.name, self._func_counter)
        self._shape_output[expout_x] = _shape
        func_list.append(ge)

        # Sum
        sumout_x = fork_name(expout_x)+"_sum"
        gr = generate_reduction("Sum", n.name, expout_x, sumout_x,
                                [1], True, self._graph.name, self._func_counter)
        self._shape_output[sumout_x] = [_shape[0], 1]
        func_list.append(gr)

        # Log
        logout_x = fork_name(sumout_x)+"_log"
        ge = generate_unary("Log", n.name, sumout_x, logout_x,
                            self._graph.name, self._func_counter)
        self._shape_output[logout_x] = [_shape[0], 1]
        func_list.append(ge)

        # Add2
        add2out_x = fork_name(rout_x)+"_add2"
        ga = generate_arithmetic("Add2", n.name, [mout_x, logout_x], add2out_x,
                                 self._graph.name, self._func_counter)
        self._shape_output[add2out_x] = [_shape[0], 1]
        func_list.append(ga)

        # Sub2
        sub2out = fork_name(n.output[0])+"sub2"
        ga = generate_arithmetic("Sub2", n.name, [rout_x, add2out_x], sub2out,
                                 self._graph.name, self._func_counter)
        self._shape_output[sub2out] = _shape
        func_list.append(ga)

        # Reshape
        rp = generate_reshape(n.name, sub2out, n.output[0], input_shape,
                              self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = input_shape
        func_list.append(rp)

    def Clip(self, opset, func_list, n):
        func = self.generate_default_function("Clip", n)
        maxval = None
        minval = None
        if opset == 6:
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
        else:
            input_num = len(n.input)
            if input_num == 1:
                pass
            elif input_num == 2:
                if n.input[1]:
                    minval = float(self.get_input_raw_data(
                        n.input[1], TensorProto.FLOAT)[0])
            elif input_num == 3:
                if n.input[1]:
                    minval = float(self.get_input_raw_data(
                        n.input[1], TensorProto.FLOAT)[0])
                if n.input[2]:
                    maxval = float(self.get_input_raw_data(
                        n.input[2], TensorProto.FLOAT)[0])
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
            minout = fork_name(minin)+"_min"
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
        input_shape = self.get_func_input_shape(func.input[0])
        axes = []
        for attr in n.attribute:
            if attr.name == "axes":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for axes in {} op_type".format(n.op_type))
                for index in attr.ints:
                    axes.append(index)
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        output_shape = [0] * (len(input_shape) + len(axes))
        for axis in axes:
            output_shape[axis] = 1
        for i in reversed(range(len(output_shape))):
            if output_shape[i] == 0:
                output_shape[i] = input_shape.pop()
        rp.shape.dim.extend(output_shape)
        func_list.append(func)
        self._shape_output[func.output[0]] = output_shape

    def Sqrt(self, func_list, n):
        pow = generate_pow_scalar(n.name, n.input[0], n.output[0],
                                  0.5, self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = self.get_func_input_shape(n.input[0])
        func_list.append(pow)

    def Tile(self, func_list, n):
        func = self.generate_default_function("Tile", n)
        tp = func.tile_param
        repeats = func.input[1]
        input_shape = self.get_func_input_shape(func.input[0])
        output_shape = []
        raw_data = self.get_input_raw_data(repeats, TensorProto.INT64)
        tp.reps.extend(raw_data)
        del func.input[1]
        if len(tp.reps) > len(input_shape):
            s = len(tp.reps) - len(input_shape)
            output_shape.extend(tp.reps[:s])
            for i, v in enumerate(input_shape):
                output_shape.append(v * tp.reps[s + i])
        else:
            s = len(input_shape) - len(tp.reps)
            output_shape.extend(input_shape[:s])
            for i, v in enumerate(tp.reps):
                output_shape.append(v * input_shape[s + i])
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Slice(self, opset, func_list, n):
        func = self.generate_default_function("Slice", n)
        sp = func.slice_param
        input_shape = self.get_func_input_shape(func.input[0])
        axes = []
        starts = []
        ends = []
        steps = []
        if opset == '6':
            for attr in n.attribute:
                if attr.name == "axes":
                    if attr.type != AttributeProto.INTS:
                        raise ValueError(
                            "Only INTS is supported for axes in {} op_type".format(n.op_type))
                    for index in attr.ints:
                        axes.append(index)
                elif attr.name == "starts":
                    if attr.type != AttributeProto.INTS:
                        raise ValueError(
                            "Only INTS is supported for starts in {} op_type".format(n.op_type))
                    for index in attr.ints:
                        starts.append(index)
                elif attr.name == "ends":
                    if attr.type != AttributeProto.INTS:
                        raise ValueError(
                            "Only INTS is supported for ends in {} op_type".format(n.op_type))
                    for index in attr.ints:
                        ends.append(index)
                else:
                    logger.info('Unsupported attribute {} was specified at {}'
                                .format(attr.name, n.op_type))
        else:
            del func.input[1:]
            starts = self.get_input_raw_data(n.input[1], TensorProto.INT64)
            ends = self.get_input_raw_data(n.input[2], TensorProto.INT64)
            try:
                axes = self.get_input_raw_data(n.input[3], TensorProto.INT64)
            except:
                pass
            try:
                steps = self.get_input_raw_data(n.input[4], TensorProto.INT64)
            except:
                pass

        output_shape = []
        if len(axes) == 0:
            axes = list(range(len(input_shape)))
        else:
            axes = [len(input_shape) + axis if axis <
                    0 else axis for axis in axes]
        if len(steps) == 0:
            steps = [1] * len(input_shape)
        else:
            steps = [1] * (len(input_shape) - len(steps)) + steps
        for i in range(len(input_shape)):
            if i not in axes:
                starts.insert(i, 0)
                ends.insert(i, input_shape[i])
            if ends[i] > input_shape[i]:
                ends[i] = input_shape[i]
            if starts[i] > input_shape[i]:
                starts[i] = input_shape[i]
            if ends[i] < 0:
                ends[i] = input_shape[i] + ends[i]
            if starts[i] < 0:
                starts[i] = input_shape[i] + ends[i]
            if starts[i] < ends[i]:
                output_shape.append(starts[i] - ends[i])
            else:
                output_shape.append(0)
        sp.start.extend(starts)
        sp.stop.extend(ends)
        sp.step.extend(steps)
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Flatten(self, func_list, n):
        # Convert to Reshape
        func = self.generate_default_function("Reshape", n)
        rp = func.reshape_param
        input_shape = self.get_func_input_shape(func.input[0])
        axis = 1
        output_shape = [1, 1]
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for axis in {} op_type".format(n.op_type))
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))

        for i in range(axis):
            output_shape[0] *= input_shape[i]
        for i in range(axis, len(input_shape)):
            output_shape[1] *= input_shape[i]
        rp.shape.dim.extend(output_shape)
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Squeeze(self, func_list, n):
        # Convert to Reshape
        func = self.generate_default_function("Reshape", n)
        output_shape = self.get_func_input_shape(func.input[0])
        rp = func.reshape_param
        axes = []
        for attr in n.attribute:
            if attr.name == "axes":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for axes in {} op_type".format(n.op_type))
                axes.extend([len(output_shape) + i if i <
                             0 else i for i in attr.ints])
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))

        if len(axes):
            output_shape = [output_shape[i]
                            for i in range(len(output_shape)) if i not in axes]
        else:
            output_shape = [i for i in output_shape if i != 1]
        rp.shape.dim.extend(output_shape)
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def DepthToSpace(self, func_list, n):
        # Convert to Reshape+Transpose+Reshape
        b, c, h, w = self.get_func_input_shape(n.input[0])
        blocksize = None
        mode = "DCR"
        for attr in n.attribute:
            if attr.name == "blocksize":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for blocksize in {} op_type".format(n.op_type))
                blocksize = attr.i
            elif attr.name == "mode":
                if attr.type != AttributeProto.STRING:
                    raise ValueError(
                        "Only STRING is supported for mode in {} op_type".format(n.op_type))
                mode = attr.s.decode("utf-8")
            else:
                raise ValueError('Unsupported attribute {} was specified at {}'
                                 .format(attr.name, n.op_type))

        if blocksize is None:
            raise ValueError("Missing 'blocksize' attribute")

        # Reshape
        rin = n.input[0]
        rout = fork_name(n.input[0])+"_reshape"
        if mode == "DCR":
            _shape = [b, blocksize, blocksize, c // (blocksize**2), h, w]
        else:
            _shape = [b, c // (blocksize**2), blocksize, blocksize, h, w]
        rp = generate_reshape(n.name, rin, rout, _shape,
                              self._graph.name, self._func_counter)
        self._shape_output[rout] = _shape
        func_list.append(rp)

        # Transpose
        trans_out = fork_name(rout)+"_trans"
        if mode == "DCR":
            axes = [0, 3, 4, 1, 5, 2]
        else:
            axes = [0, 1, 4, 2, 5, 3]
        transp = generate_transpose(n.name, rout, trans_out,
                                    axes, self._graph.name, self._func_counter)
        output_shape = []
        for i in range(len(_shape)):
            index = axes[i]
            output_shape.append(_shape[index])
        self._shape_output[trans_out] = output_shape
        func_list.append(transp)

        # Reshape
        _shape = [b, c // (blocksize**2), h * blocksize, w * blocksize]
        rp = generate_reshape(n.name, trans_out, n.output[0], _shape,
                              self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = _shape
        func_list.append(rp)

    def SpaceToDepth(self, func_list, n):
        # Convert to Reshape+Transpose+Reshape
        b, c, h, w = self.get_func_input_shape(n.input[0])
        blocksize = None
        for attr in n.attribute:
            if attr.name == "blocksize":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for blocksize in {} op_type".format(n.op_type))
                blocksize = attr.i
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))

        if blocksize is None:
            raise ValueError("Missing 'blocksize' attribute")

        reduced_h = h // blocksize
        reduced_w = w // blocksize
        # Reshape
        rin = n.input[0]
        rout = fork_name(n.input[0])+"_reshape"
        _shape = [b, c, reduced_h, blocksize, reduced_w, blocksize]
        rp = generate_reshape(n.name, rin, rout, _shape,
                              self._graph.name, self._func_counter)
        self._shape_output[rout] = _shape
        func_list.append(rp)

        # Transpose
        trans_out = fork_name(rout)+"_trans"
        axes = [0, 3, 5, 1, 2, 4]
        transp = generate_transpose(n.name, rout, trans_out,
                                    axes, self._graph.name, self._func_counter)
        output_shape = []
        for i in range(len(_shape)):
            index = axes[i]
            output_shape.append(_shape[index])
        self._shape_output[trans_out] = output_shape
        func_list.append(transp)

        # Reshape
        _shape = [b, c * (blocksize**2), reduced_h, reduced_w]
        rp = generate_reshape(n.name, trans_out, n.output[0], _shape,
                              self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = _shape
        func_list.append(rp)

    def ElementIndices(self, func_name, func_list, n):
        # Convert to Max or Min
        func = self.generate_default_function(func_name, n)
        output_shape = self.get_func_input_shape(func.input[0])
        if func_name == "Max":
            mp = func.max_param
        else:
            mp = func.min_param
        mp.only_index = True
        mp.keep_dims = True
        axes = [0]
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for axis in {} op_type".format(n.op_type))
                if attr.i < 0:
                    axes = [len(output_shape) + attr.i]
                else:
                    axes = [attr.i]
            elif attr.name == "keepdims":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for keepdims in {} op_type".format(n.op_type))
                mp.keep_dims = bool(attr.i)
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))
        mp.axes.extend(axes)
        for i in mp.axes:
            if mp.keep_dims:
                output_shape[i] = 1
            else:
                del output_shape[i]
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def Split(self, func_list, n):
        # Convert to Split+Stack
        input_shape = self.get_func_input_shape(n.input[0])
        axis = 0
        offset = 0
        output_len = []
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for axis in {} op_type".format(n.op_type))
                axis = attr.i
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i
            elif attr.name == "split":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for split in {} op_type".format(n.op_type))
                output_len.extend(attr.ints)
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))

        if len(output_len) == 0:
            output_len = [input_shape[axis] // len(n.output)] * len(n.output)

        # Split
        sout = []
        for i in range(input_shape[axis]):
            sout.append(n.input[0]+"_split_"+str(i))
        sp = generate_split(n.name, n.input[0], sout, axis,
                            self._graph.name, self._func_counter)
        func_list.append(sp)

        for i in range(len(sout)):
            self._shape_output[sout[i]] = [input_shape[x]
                                           for x in range(len(input_shape)) if x != axis]

        # Stack
        for i in range(len(n.output)):
            shape = [input_shape[x] if x != axis else output_len[i]
                     for x in range(len(input_shape))]
            sp = generate_stack(n.name, sout[offset:output_len[i]+offset], n.output[i], axis,
                                self._graph.name, self._func_counter)
            self._shape_output[n.output[i]] = shape
            func_list.append(sp)
            offset += output_len[i]

    def Upsample_6(self, func_list, n):
        func = self.generate_default_function("Unpooling", n)
        input_shape = self.get_func_input_shape(n.input[0])
        upp = func.unpooling_param
        scales = [1, 1]
        for attr in n.attribute:
            if attr.name == "height_scale":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for height_scale in {} op_type".format(n.op_type))
                scales[0] = int(np.floor(attr.f))
            elif attr.name == "width_scale":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for width_scale in {} op_type".format(n.op_type))
                scales[1] = int(np.floor(attr.f))
            elif attr.name == "mode":
                pass
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))
        scales = [1] * (len(input_shape) - 2) + scales
        output_shape = []
        for i in range(len(input_shape)):
            output_shape.append(input_shape[i] * scales[i])
        self._shape_output[n.output[0]] = output_shape
        upp.kernel.dim.extend(scales)
        func_list.append(func)

    def Upsample_7(self, func_list, n):
        func = self.generate_default_function("Unpooling", n)
        input_shape = self.get_func_input_shape(n.input[0])
        upp = func.unpooling_param
        scales = []
        for attr in n.attribute:
            if attr.name == "scales":
                if attr.type != AttributeProto.FLOATS:
                    raise ValueError(
                        "Only FLOATS is supported for scales in {} op_type".format(n.op_type))
                scales.extend([int(np.floor(f)) for f in attr.floats])
            elif attr.name == "mode":
                pass
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))

        if len(scales) == 0:
            raise ValueError("Missing 'scales' attribute")

        output_shape = []
        for i in range(len(input_shape)):
            output_shape.append(input_shape[i] * scales[i])
        self._shape_output[n.output[0]] = output_shape
        upp.kernel.dim.extend(scales)
        func_list.append(func)

    def Upsample_9(self, func_list, n):
        func = self.generate_default_function("Unpooling", n)
        input_shape = self.get_func_input_shape(n.input[0])
        upp = func.unpooling_param
        scales = []
        for init in self._graph.initializer:
            if init.name == n.input[1]:
                if init.data_type != TensorProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for {} in {} op_type".format(n.input[1], n.op_type))
                if init.raw_data:
                    scales.extend(np.fromstring(
                        init.raw_data, dtype=np.float32))
                elif init.float_data:
                    scales.extend(init.float_data)
        self._merged_inputs.append(n.input[1])
        del func.input[1]
        scales = [int(np.floor(i)) for i in scales]
        output_shape = []
        for i in range(len(input_shape)):
            output_shape.append(input_shape[i] * scales[i])
        self._shape_output[n.output[0]] = output_shape
        upp.kernel.dim.extend(scales)
        func_list.append(func)

    def Mean(self, func_list, n):
        # Reshape+Broadcast+Stack+Mean
        func = self.generate_default_function("Mean", n)
        input_num = len(func.input)
        if input_num == 1:
            func.type = "Identity"
            func_list.append(func)
            return
        else:
            input_shape = []
            ndim = 0
            inputs = func.input[:]
            for name in inputs:
                shape = self.get_func_input_shape(name)
                if len(shape) > ndim:
                    ndim = len(shape)
                input_shape.append(shape)
                func.input.remove(name)

            # In opset>8 version, `Mean` support multi-directional broadcast,
            # So all the input is processed into the same shape as the output through Reshape and broadcast operations.
            # But caffe2 does not support multi-directional broadcast.
            for i in range(len(input_shape)):
                if len(input_shape[i]) < ndim:
                    input_shape[i] = list(
                        np.ones(ndim - len(input_shape[i]), dtype=np.int)) + input_shape[i]
                    rout = fork_name(inputs[i])+"_shape"
                    rp = generate_reshape(n.name, inputs[i], rout, input_shape[i],
                                          self._graph.name, self._func_counter)
                    self._shape_output[rout] = input_shape[i]
                    func_list.append(rp)
                    inputs[i] = rout

            broadcast_shape = list(np.max(input_shape, axis=0))
            for i in range(len(input_shape)):
                need_broadcast = False
                for j in range(ndim):
                    if input_shape[i][j] != broadcast_shape[j]:
                        need_broadcast = True
                        break
                if need_broadcast:
                    bout = fork_name(inputs[i]) + "_broadcast"
                    bt = generate_broadcast(n.name, inputs[i], bout, broadcast_shape,
                                            self._graph.name, self._func_counter)
                    self._shape_output[bout] = shape
                    inputs[i] = bout
                    func_list.append(bt)

            sout = fork_name(func.output[0]) + "_stack"
            sp = generate_stack(n.name, inputs, sout, 0,
                                self._graph.name, self._func_counter)
            self._shape_output[sout] = [len(input_shape)] + broadcast_shape
            func_list.append(sp)

            func.input.extend([sout])
            mp = func.mean_param
            mp.axes.extend([0])
            func_list.append(func)

    def ConvTranspose(self, func_list, n):
        func = self.generate_default_function("Deconvolution", n)
        cp = func.deconvolution_param
        input_shape = self.get_func_input_shape(func.input[0])
        weight_shape = self.get_func_input_shape(func.input[1])
        # We shouldn't need these default settings
        # since NNabla will set these for us
        cp.base_axis = 1
        cp.group = 1
        dim = len(input_shape) - 2
        auto_pad = 'NOTSET'
        pads = [0] * dim * 2
        strides = [1] * dim
        dilations = [1] * dim
        output_padding = [0] * dim
        output_shape = []
        convt_output_shape = []  # explicitly set the shape of the output.

        for attr in n.attribute:
            if attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for pads in ConvTranspose op_type")
                pads.clear()
                pads.extend(attr.ints)
            elif attr.name == "strides":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for strides in ConvTranspose op_type")
                strides.clear()
                strides.extend(attr.ints)
            elif attr.name == "dilations":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for dilations in ConvTranspose op_type")
                dilations.clear()
                dilations.extend(attr.ints)
            elif attr.name == "group":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for group in ConvTranspose op_type")
                cp.group = attr.i
            elif attr.name == "kernel_shape":
                # We do not set 'kernel_shape' to NNabla
                # since NNabla doesn't have a parameter for it
                # (it will be inferred from weight input)
                pass
            elif attr.name == "output_padding":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for dilations in ConvTranspose op_type")
                output_padding.clear()
                output_padding.extend(attr.ints)
            elif attr.name == "output_shape":
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS are supported for dilations in ConvTranspose op_type")
                convt_output_shape.extend(attr.ints)
            elif attr.name == "auto_pad":
                if attr.type != AttributeProto.STRING:
                    raise ValueError(
                        "Only STRING is supported for auto_pad in ConvTranspose op_type")
                auto_pad = attr.s.decode("utf-8")
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        # NNabla requires for the dimensions of strides, pads, dilations to match.
        # We align the dimensions for all three attributes to the shortest one
        if strides:
            cp.stride.dim.extend(strides[:])
        else:
            cp.stride.dim.extend([1]*dim)
        if dilations:
            cp.dilation.dim.extend(dilations[:])
        else:
            # Set default values.
            # Do we really need this? (Default value should be set by NNabla)
            cp.dilation.dim.extend([1]*dim)

        if convt_output_shape:
            for index in range(dim):
                d = cp.dilation.dim[index]
                s = cp.stride.dim[index]
                w = weight_shape[2 + index]
                i = input_shape[2 + index]
                o = convt_output_shape[index]
                adj = output_padding[index]
                p = (i - 1) * s + adj + (w - 1) * d + 1 - o
                if p <= 0:
                    output_padding[index] -= p
                else:
                    if auto_pad == 'SAME_UPPER':
                        pads[index] = p - int(p / 2)
                        pads[index+dim] = int(p / 2)
                    else:
                        pads[index] = int(p / 2)
                        pads[index + dim] = p - int(p / 2)
            output_shape = convt_output_shape
        else:
            for index in range(dim):
                d = cp.dilation.dim[index]
                s = cp.stride.dim[index]
                w = weight_shape[2 + index]
                i = input_shape[2 + index]
                adj = output_padding[index]
                o = (i - 1) * s + adj + (w - 1) * \
                    d + 1 - pads[index] - pads[index+dim]
                output_shape.append(o)

        padval = []
        asymmetry = check_padding(pads, dim, padval)
        if asymmetry:
            # Add a separate padding function for
            # asymmetry padding
            input = n.input[0]
            padded = fork_name(input) + "_pad"
            pad_width = rearrange_pads(pads)
            padf = generate_pad(n.name, input, padded,
                                "constant", pad_width, 0,
                                self._graph.name, self._func_counter)
            shape = input_shape[:]
            for i in range(dim):
                shape[i + 2] += pad_width[2 * i]
                shape[i + 2] += pad_width[2 * i + 1]
            self._shape_output[padded] = shape
            func_list.append(padf)
            func.input[0] = padded
        cp.pad.dim.extend(padval)

        if output_padding != [0] * dim:
            deconv_out = fork_name(func.output[0]) + "_deconv"
            del func.output[:]
            func.output.extend([deconv_out])
            self._shape_output[deconv_out] = input_shape[:1] + \
                [weight_shape[1]] + [output_shape[i] - output_padding[i]
                                     for i in range(dim)]
            func_list.append(func)

            pad_width = [output_padding[i % 2] if i %
                         2 else 0 for i in range(2*dim)]
            padf = generate_pad(n.name, deconv_out, n.output[0],
                                "constant", pad_width, 0,
                                self._graph.name, self._func_counter)
            func_list.append(padf)
        else:
            func_list.append(func)
        self._shape_output[n.output[0]] = input_shape[:1] + \
            [weight_shape[1]] + output_shape

    def ConstantOfShape(self, func_list, n):
        func = self.generate_default_function("Constant", n)
        cp = func.constant_param
        cp.val = 0.0
        for attr in n.attribute:
            if attr.name == "value":
                if attr.t.data_type == TensorProto.FLOAT:
                    cp.val = attr.t.float_data[0]
                elif attr.t.data_type == TensorProto.INT32:
                    cp.val = attr.t.int32_data[0]
                elif attr.t.data_type == TensorProto.INT64:
                    cp.val = attr.t.int64_data[0]
                else:
                    raise ValueError(
                        "Unsupported tensor data type: {}".format(attr.t.data_type))

        raw_data = self.get_input_raw_data(func.input[0], TensorProto.INT64)
        cp.shape.dim.extend(raw_data)
        del func.input[0]
        self._shape_output[func.output[0]] = raw_data
        func_list.append(func)

    def Expand(self, func_list, n):
        func = self.generate_default_function("Broadcast", n)
        bp = func.broadcast_param
        input = n.input[0]

        input_shape = self.get_func_input_shape(n.input[0])

        raw_data = self.get_input_raw_data(func.input[1], TensorProto.INT64)

        if len(raw_data) > len(input_shape):
            new_shape = [1] * (len(raw_data) - len(input_shape)) + input_shape
            rout = fork_name(n.input[0])+"_shape"
            rp = generate_reshape(n.name, n.input[0], rout, new_shape,
                                  self._graph.name, self._func_counter)
            self._shape_output[rout] = new_shape
            func_list.append(rp)
            input = rout

        bp.shape.dim.extend(raw_data)

        del func.input[1]
        func.input[0] = input
        self._shape_output[func.output[0]] = raw_data
        func_list.append(func)

    def HardSigmoid(self, func_list, n):
        alpha = 0.2
        beta = 0.5
        shape = self.get_func_input_shape(n.input[0])
        for attr in n.attribute:
            if attr.name == "alpha":
                alpha = attr.f
            elif attr.name == "beta":
                beta = attr.f
        if alpha == 0.2 and beta == 0.5:
            func = self.generate_default_function("HardSigmoid", n)
            func_list.append(func)
        else:
            muls_out = fork_name(n.input[0])+"_muls"
            muls = generate_mul_scalar(n.name, n.input[0], muls_out,
                                       alpha, self._graph.name, self._func_counter)
            self._shape_output[muls_out] = shape
            func_list.append(muls)

            adds_out = fork_name(n.input[0])+"_adds"
            adds = generate_add_scalar(n.name, muls_out, adds_out,
                                       beta, self._graph.name, self._func_counter)
            self._shape_output[adds_out] = shape
            func_list.append(adds)

            mins_out = fork_name(n.input[0])+"_mins"
            mins = generate_minimum_scalar(n.name, adds_out, mins_out,
                                           1, self._graph.name, self._func_counter)
            self._shape_output[mins_out] = shape
            func_list.append(mins)

            maxs = generate_maximum_scalar(n.name, mins_out, n.output[0],
                                           0, self._graph.name, self._func_counter)
            self._shape_output[n.output[0]] = shape
            func_list.append(maxs)

    def Hardmax(self, func_list, n):
        axis = 1
        input_shape = self.get_func_input_shape(n.input[0])
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i
        new_shape = [int(np.prod(input_shape[:axis])),
                     int(np.prod(input_shape[axis:]))]
        rout = fork_name(n.input[0])+"_shape"
        rp = generate_reshape(n.name, n.input[0], rout, new_shape,
                              self._graph.name, self._func_counter)
        self._shape_output[rout] = new_shape
        func_list.append(rp)

        max_func = self.generate_default_function("Max", n)
        max_out = fork_name(n.input[0])+"_max"
        max_func.input[0] = rout
        max_func.output[0] = max_out
        mp = max_func.max_param
        mp.only_index = True
        mp.keep_dims = True
        mp.axes.extend([1])
        self._shape_output[max_out] = [new_shape[0], 1]
        func_list.append(max_func)

        one_hot_func = self.generate_default_function("OneHot", n)
        one_hot_out = fork_name(n.input[0])+"_one_hot"
        one_hot_func.input[0] = max_out
        one_hot_func.output[0] = one_hot_out
        one_hot_p = one_hot_func.one_hot_param
        one_hot_p.shape.dim.extend([new_shape[1]])
        self._shape_output[one_hot_out] = new_shape
        func_list.append(one_hot_func)

        rp = generate_reshape(n.name, one_hot_out, n.output[0], input_shape,
                              self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = input_shape
        func_list.append(rp)

    def InstanceNormalization(self, func_list, n):
        input_shape = self.get_func_input_shape(n.input[0])
        scale_shape = self.get_func_input_shape(n.input[1])
        nnp_order = [0, 2, 1]
        nnp_input = [n.input[i] for i in nnp_order]

        mean_out = fork_name(n.input[0])+"_mean"
        mean_param = create_parameter_variable(self._pb, mean_out,
                                               scale_shape, [0]*scale_shape[0])
        self._param_vars[mean_out] = None
        nnp_input.append(mean_out)
        variance_out = fork_name(n.input[0])+"_variance"
        variance_param = create_parameter_variable(self._pb, variance_out,
                                                   scale_shape, [0]*scale_shape[0])
        self._param_vars[variance_out] = None
        nnp_input.append(variance_out)

        epsilon = 1e-05
        for attr in n.attribute:
            if attr.name == "epsilon":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError(
                        "Only FLOAT is supported for epsilon in InstanceNormalization op_type")
                epsilon = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        if input_shape[0] > 1:
            # Split
            sout = []
            for i in range(input_shape[0]):
                sout.append(n.input[0]+"_split_"+str(i))
                self._shape_output[n.input[0] +
                                   "_split_"+str(i)] = input_shape[1:]
            sp = generate_split(n.name, n.input[0], sout, 0,
                                self._graph.name, self._func_counter)
            func_list.append(sp)

            # Reshape
            for i in range(len(sout)):
                rout = sout[i]+"_reshape"
                rp = generate_reshape(n.name, sout[i], rout, [1] + input_shape[1:],
                                      self._graph.name, self._func_counter)
                sout[i] = rout
                func_list.append(rp)
                self._shape_output[rout] = [1] + input_shape[1:]

            # BatchNormalization
            all_bn_out = []
            for i in range(len(sout)):
                bn_out = n.input[0]+"_bn_"+str(i)
                all_bn_out.append(bn_out)
                bn_func = self.generate_default_function(
                    "BatchNormalization", n)
                del bn_func.input[:]
                bn_func.input.extend(nnp_input)
                bnp = bn_func.batch_normalization_param
                bnp.axes.extend([1])
                bnp.eps = epsilon
                bnp.decay_rate = 0.9
                bnp.batch_stat = True
                bn_func.input[0] = sout[i]
                bn_func.output[0] = bn_out
                func_list.append(bn_func)
                self._shape_output[bn_out] = [1] + input_shape[1:]

            # Concatenate
            concatenate_func = self.generate_default_function("Concatenate", n)
            del concatenate_func.input[:]
            concatenate_func.input.extend(all_bn_out)
            concatenate_func.output[0] = n.output[0]
            concatenate_p = concatenate_func.concatenate_param
            concatenate_p.axis = 0
            self._shape_output[n.output[0]] = input_shape
            func_list.append(concatenate_func)
        else:
            bn_func = self.generate_default_function("BatchNormalization", n)
            del bn_func.input[:]
            bn_func.input.extend(nnp_input)
            bnp = bn_func.batch_normalization_param
            bnp.axes.extend([1])
            bnp.eps = epsilon
            bnp.decay_rate = 0.9
            bnp.batch_stat = True
            func_list.append(bn_func)
            self._shape_output[n.output[0]] = input_shape
        new_const_inp_shape = [1] * len(input_shape)
        new_const_inp_shape[1] = scale_shape[0]
        for inp in nnp_input[1:5]:
            self._parameter_shape[inp] = new_const_inp_shape

    def ThresholdedRelu(self, func_list, n):
        alpha = 1.0
        for attr in n.attribute:
            if attr.name == "alpha":
                alpha = attr.f

        input_shape = self.get_func_input_shape(n.input[0])
        cout = fork_name(n.input[0])+"_zeros"
        constant_func = self.generate_default_function("Constant", n)
        cp = constant_func.constant_param
        cp.val = 0.0
        cp.shape.dim.extend(input_shape)
        del constant_func.input[0]
        constant_func.output[0] = cout
        func_list.append(constant_func)
        self._shape_output[cout] = input_shape

        gout = fork_name(n.input[0])+"_greaterscalar"
        greater_func = self.generate_default_function("GreaterScalar", n)
        gp = greater_func.greater_scalar_param
        gp.val = alpha
        greater_func.output[0] = gout
        func_list.append(greater_func)
        self._shape_output[gout] = input_shape

        where_func = self.generate_default_function("Where", n)
        del where_func.input[:]
        where_func.input.extend([gout, n.input[0], cout])
        func_list.append(where_func)
        self._shape_output[n.output[0]] = input_shape

    def ReduceSumSquare(self, func_list, n):
        input_shape = self.get_func_input_shape(n.input[0])
        keepdims = 1
        axes = []
        for attr in n.attribute:
            if attr.name == "axes":
                axes.extend(attr.ints)
            elif attr.name == "keepdims":
                keepdims = attr.i

        if len(axes):
            axes = [len(input_shape) + axis if axis <
                    0 else axis for axis in axes]
        else:
            axes = list(range(len(input_shape)))

        output_shape = input_shape[:]
        for i in axes:
            if keepdims:
                output_shape[i] = 1
            else:
                del output_shape[i]

        # PowScalar
        pow_scalar_out = fork_name(n.input[0])+"_pow2"
        pow_func = generate_pow_scalar(n.name, n.input[0], pow_scalar_out,
                                       2, self._graph.name, self._func_counter)
        self._shape_output[pow_scalar_out] = input_shape
        func_list.append(pow_func)

        sum_func = generate_reduction("Sum", n.name, pow_scalar_out, n.output[0],
                                      axes, keepdims, self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = output_shape
        func_list.append(sum_func)

    def Cast(self, func_list, n):
        # Save Cast Node Information, No need to add any function.
        self._cast_node[n.output[0]] = n.input[0]
        self._removed_outputs.append(n.output[0])

    def Gather(self, func_list, n):
        # Convert to Slice + Concatenate
        input_shape = self.get_func_input_shape(n.input[0])
        axis = 0
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i

        indices = self.get_input_raw_data(n.input[1], TensorProto.INT64)
        indices = [input_shape[axis] + index if index <
                   0 else index for index in indices]

        slice_outs = []
        start = [0] * len(input_shape)
        stop = input_shape[:]
        step = [1] * len(input_shape)
        for index in indices:
            start[axis] = index
            stop[axis] = index + 1
            slice_out = fork_name(n.input[0])+"_slice_"+str(index)
            func = self.generate_default_function("Slice", n)
            del func.input[1]
            func.output[0] = slice_out
            sp = func.slice_param
            sp.start.extend(start)
            sp.stop.extend(stop)
            sp.step.extend(step)
            slice_outs.append(slice_out)
            self._shape_output[slice_out] = [
                1 if index == axis else i for index, i in enumerate(input_shape)]
            func_list.append(func)

        func = self.generate_default_function("Concatenate", n)
        del func.input[:]
        func.input.extend(slice_outs)
        concatenate_p = func.concatenate_param
        concatenate_p.axis = axis
        self._shape_output[n.output[0]] = [
            len(indices) if index == axis else i for index, i in enumerate(input_shape)]
        func_list.append(func)

    def QuantizeLinear(self, func_list, n):
        TENSOR_TYPE_TO_INT = {
            TensorProto.INT8: 1,
            TensorProto.UINT8: 2,
        }
        func = self.generate_default_function("QuantizeLinear", n)
        qlp = func.quantize_linear_param
        qlp.round_mode = "HALF_TO_EVEN"
        qlp.narrow_range = False
        input_shape = self.get_func_input_shape(n.input[0])
        axis = 1
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i

        if len(n.input) < 2:
            zero_point = fork_name(n.input[0]) + "_zero_point"
            zero_point_shape = [1] * len(input_shape)
            zero_point_shape[axis] = input_shape[axis]
            create_parameter_variable(self._pb, zero_point, zero_point_shape, [
                                      0] * input_shape[axis])
            self._param_vars[zero_point] = None
            func.input.append(zero_point)
            qlp.dtype = 2
        else:
            dtype = TensorProto.UINT8
            for init in self._graph.initializer:
                if init.name == n.input[2]:
                    dtype = init.data_type
                    break
            qlp.dtype = TENSOR_TYPE_TO_INT[dtype]

        self._shape_output[n.output[0]] = input_shape
        func_list.append(func)

    def DequantizeLinear(self, func_list, n):
        func = self.generate_default_function("DequantizeLinear", n)
        input_shape = self.get_func_input_shape(n.input[0])
        axis = 1
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i

        if len(n.input) < 2:
            zero_point = fork_name(n.input[0]) + "_zero_point"
            zero_point_shape = [1] * len(input_shape)
            zero_point_shape[axis] = input_shape[axis]
            create_parameter_variable(self._pb, zero_point, zero_point_shape, [
                                      0] * input_shape[axis])
            self._param_vars[zero_point] = None
            func.input.append(zero_point)

        self._shape_output[n.output[0]] = input_shape
        func_list.append(func)

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
            for index, i in enumerate(f.input):
                if i in self._cast_node:
                    f.input[index] = self._cast_node[i]
                else:
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
                pass
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
        for name in list(self._param_vars.keys()):
            p = exe.parameter_variable.add()
            p.variable_name = name
        # Convert the shape of some parameters
        for var_name, shape in self._parameter_shape.items():
            for v in network.variable:
                if v.name == var_name:
                    del v.shape.dim[:]
                    v.shape.dim.extend(shape)
                    break
            for p in pb.parameter:
                if p.variable_name == var_name:
                    del p.shape.dim[:]
                    p.shape.dim.extend(shape)

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
