# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
from functools import partial
from struct import pack, unpack

import nnabla.logger as logger
import numpy as np
from nnabla.utils import nnabla_pb2

try:
    from onnx import (ModelProto, TensorProto, AttributeProto, TensorShapeProto,
                      mapping)
except:
    print('ONNX import support disabled because onnx python package is not found.')
    print(' You may install onnx package with "pip install onnx".')

from .utils import *


# ONNX does not have the concept of executors.
# We will add a single executor to NNP when converted from ONNX,
# and set a default name to it.
DEFAULT_EXECUTOR_NAME = "exec_0"

fork_name_number = 0


# Normalize shape from () to (1,).
def normalize_shape(shape):
    if isinstance(shape, TensorShapeProto):
        if len(shape.dim) == 0:
            shape.dim.extend([TensorShapeProto.Dimension(dim_value=1)])
    if isinstance(shape, list):
        if len(shape) == 0:
            shape = [1]
    return shape


def bit_cast_f32_to_i32(value: float):
    return unpack('i', pack('f', value))[0]


def add_value_info_as_variable(network, info):
    if not info.type.HasField("tensor_type"):  # accepting only tensor
        raise ValueError("Only TensorProto is allowed as ValueInfoProto's type for info.name (Got {})"
                         .format(info.name, info.type))
    t = info.type.tensor_type
    v = network.variable.add()
    v.name = info.name
    shape = normalize_shape(t.shape)
    v.shape.dim.extend(
        [x.dim_value if not x.dim_param else -1 for x in shape.dim])
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


def get_transpose_output_shape(input_shape, axes):
    i_shape = input_shape
    o_shape = []
    for i in range(len(i_shape)):
        index = axes[i]
        o_shape.append(i_shape[index])
    return o_shape


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


def generate_slice(node_name, in_name, out_name, start, stop, step, base_name, func_counter):
    """Generate a Slice operator to slice the specified buffer.
    """
    slice = nnabla_pb2.Function()
    slice.type = "Slice"
    set_function_name(slice, node_name, base_name, func_counter)
    slice.input.extend([in_name])
    slice.output.extend([out_name])
    sp = slice.slice_param
    sp.start.extend(start)
    sp.stop.extend(stop)
    sp.step.extend(step)
    return slice


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


def generate_less_scalar(node_name, x, out_name,
                         val, base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "LessScalar"
    set_function_name(func, node_name, base_name, func_counter)
    func.input.extend([x])
    func.output.extend([out_name])
    lsp = func.less_scalar_param
    lsp.val = val
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


def generate_rand_normal(node_name, out_name, mean, scale, seed, shape,
                         base_name, func_counter):
    func = nnabla_pb2.Function()
    func.type = "Randn"
    set_function_name(func, node_name, base_name, func_counter)
    func.output.extend([out_name])
    rp = func.randn_param
    rp.mu = mean
    rp.sigma = scale
    rp.seed = seed
    rp.shape.dim.extend(shape)
    return func


def generate_rand_uniform(node_name, out_name, dtype, high, low, seed, shape,
                          base_name, func_counter):
    assert low < high
    type_info = np.finfo(mapping.TENSOR_TYPE_TO_NP_TYPE[dtype])
    assert low >= type_info.min
    assert high <= type_info.max
    func = nnabla_pb2.Function()
    func.type = "Rand"
    set_function_name(func, node_name, base_name, func_counter)
    func.output.extend([out_name])
    rp = func.rand_param
    rp.high = high
    rp.low = low
    rp.seed = seed
    rp.shape.dim.extend(shape)
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


def unsupported_attribute(attr_name, n):
    raise ValueError("Unsupported attribute {} was specified at {}"
                     .format(attr_name, n.op_type))


def check_attr_int_type(attr, node):
    if attr.type != AttributeProto.INT:
        raise ValueError(
            f"Only INT is supported for {attr.name} in {node.op_type} op_type")


def check_attr_string_type(attr, node):
    if attr.type != AttributeProto.STRING:
        raise ValueError(
            f"Only STRING is supported for {attr.name} in {node.op_type} op_type")


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
    shape = normalize_shape(tensor.dims)
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
            p.data.extend(np.fromstring(tensor.raw_data, dtype=bool))
        elif tensor.int32_data:
            p.data.extend(tensor.int32_data)
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
            "GlobalMaxPool": self.GlobalMaxPool,
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
            "Softplus": self.Softplus,
            "Softsign": partial(self.GeneralOperator, 'SoftSign'),
            "LRN": self.LRN,
            "Clip": partial(self.Clip, 6),
            # Constant does not get converted to a function
            # but we list it here so we can accept it
            "Constant": self.Constant,
            "Unsqueeze": partial(self.Unsqueeze, '6'),
            "Sqrt": self.Sqrt,
            "Ceil": partial(self.GeneralOperator, 'Ceil'),
            "Floor": partial(self.GeneralOperator, 'Floor'),
            "Tile": self.Tile,
            "Flatten": self.Flatten,
            "Squeeze": partial(self.Squeeze, '6'),
            "Slice": partial(self.Slice, '6'),
            # Currently, caffe2 does not support this function.
            "DepthToSpace": self.DepthToSpace,
            "SpaceToDepth": self.SpaceToDepth,
            "ArgMax": partial(self.ElementIndices, "Max"),
            "ArgMin": partial(self.ElementIndices, "Min"),
            "Split": partial(self.Split, '6'),
            "Upsample": self.Upsample_6,
            "Mean": self.Mean,
            "ConvTranspose": self.ConvTranspose,
            "ConstantOfShape": self.ConstantOfShape,
            "HardSigmoid": self.HardSigmoid,
            "Hardmax": self.Hardmax_6,
            "InstanceNormalization": self.InstanceNormalization,
            "ReduceSumSquare": self.ReduceSumSquare,
            # Currently, Cast does not get converted to a function
            # but we list it here so we can accept it
            "Cast": self.Cast,
            "Gather": self.Gather,
            "RandomNormal": self.RandomNormal,
            "RandomNormalLike": self.RandomNormalLike,
            "RandomUniform": self.RandomUniform,
            "RandomUniformLike": self.RandomUniformLike,
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
            "Compress": self.Compress,
        }
        self.table_op_set_9 = dict(self.table_op_set_7, **self.table_op_set_9)

        # opset_10 table
        self.table_op_set_10 = {
            "ThresholdedRelu": self.ThresholdedRelu,
            "IsInf": partial(self.GeneralOperator, 'IsInf'),
            "Slice": partial(self.Slice, '10'),
            "QuantizeLinear": self.QuantizeLinear,
            "DequantizeLinear": self.DequantizeLinear,
            "TopK": self.TopK,
        }
        self.table_op_set_10 = dict(
            self.table_op_set_9, **self.table_op_set_10)

        # opset_11 table
        self.table_op_set_11 = {
            "Clip": partial(self.Clip, 11),
            "Round": self.Round,
            "Pad": partial(self.Pad, '11'),
        }
        self.table_op_set_11 = dict(
            self.table_op_set_10, **self.table_op_set_11)
        # Currently, we only planed to support opset 6 and opset 11.
        # More planes will be added later to support more opset versions.

        # opset_13 table
        self.table_op_set_13 = {
            "ReduceSum": self.ReduceSum,
            "Squeeze": partial(self.Squeeze, '13'),
            "Unsqueeze": partial(self.Unsqueeze, '13'),
            "Split": partial(self.Split, '13'),
            "Hardmax": self.Hardmax_13,
            "LessOrEqual": partial(self.BroadcastOperator_9, 'LessEqual'),
            "GreaterOrEqual": partial(self.BroadcastOperator_9, 'GreaterEqual'),
            "Celu": self.Celu,
            "Softmax": self.Softmax_13,
            "LogSoftmax": self.LogSoftmax_13,
            "Shape": self.Shape,
            "Resize": self.Resize_13,
            "ScatterElements": self.ScatterElements_13,
        }
        self.table_op_set_13 = dict(
            self.table_op_set_11, **self.table_op_set_13)

        self.opver_impl_map = {
            "6": self.table_op_set_6,
            "7": self.table_op_set_7,
            "9": self.table_op_set_9,
            "10": self.table_op_set_10,
            "11": self.table_op_set_11,
            "12": self.table_op_set_13,
            "13": self.table_op_set_13,
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
                    shape = normalize_shape(t.shape)
                    input_shape = [
                        x.dim_value if not x.dim_param else 1 for x in shape.dim]
                    return input_shape
            for i in self._graph.initializer:
                if i.name == input_name:
                    input_shape = normalize_shape(i.dims)
                    return list(input_shape)
        if not input_shape:
            raise ValueError(
                "The shape of {} was not found".format(input_name))
        return input_shape

    def get_func_input_dtype(self, input_name):
        dtype = None
        if input_name in self._cast_node:
            input_name = self._cast_node[input_name]
        if input_name in self._shape_output:
            dtype = TensorProto.FLOAT
        else:
            for i in self._graph.input:
                if i.name == input_name:
                    return i.type.tensor_type.elem_type
            for i in self._graph.initializer:
                if i.name == input_name:
                    return i.data_type
        if not dtype:
            raise ValueError(
                "The dtype of {} was not found".format(input_name))
        return dtype

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
                                    attr.t.raw_data, dtype=bool))
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
                            init.raw_data, dtype=bool))
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
            kernels = weight_shape[2:]
            pads_value = [0] * dims
            if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                for i in range(dims):
                    k = kernels[i]
                    s = strides[i]
                    d = dilations[i]
                    i_size = input_shape[2 + i]
                    o_size = int(np.ceil(i_size / s))
                    pads_value[i] = (o_size - 1) * s + d * (k - 1) - i_size + 1
            elif auto_pad == 'VALID':
                pass
            for i in range(dims):
                if auto_pad == 'SAME_LOWER':
                    pads[i + dims] = pads_value[i] // 2
                    pads[i] = pads_value[i] - pads[i + dims]
                elif auto_pad == 'SAME_UPPER':
                    pads[i] = pads_value[i] // 2
                    pads[i + dims] = pads_value[i] - pads[i]

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
        if not tp.axes:
            ts = list(range(len(input_shape)))
            ts.reverse()
            tp.axes.extend(ts)
        output_shape = []
        for i in range(len(input_shape)):
            index = tp.axes[i]
            output_shape.append(input_shape[index])
        self._shape_output[func.output[0]] = output_shape
        func_list.append(func)

    def BasePooling(self, func_name, func_list, n):
        def _compute_output_size(insize, stride, kernel, ceil_mode, pad_needed):
            if not pad_needed:
                pad_needed = [0] * len(insize)
            out_shape = [0] * len(insize)
            out_shape_ceil = [0] * len(insize)
            for index in range(len(insize)):
                i = insize[index]
                s = stride[index]
                k = kernel[index]
                p = pad_needed[index]
                out_shape[index] = int(np.floor((i + p - k) / s + 1))
                out_shape_ceil[index] = int(np.ceil((i + p - k) / s + 1))
            if ceil_mode != 0 and out_shape != out_shape_ceil:
                raise ValueError(
                    "Unsupported attribute ceil_mode=1 was specified at {}"
                    .format(n.op_type))
            return out_shape

        def _compute_pad_needed(insize, stride, kernel):
            pad_needed = [0] * len(insize)
            for index in range(len(insize)):
                s = stride[index]
                k = kernel[index]
                i = insize[index]
                target_output_shape = int((i + s - 1) / s)
                pad_needed[index] = int((target_output_shape - 1) * s + k - i)
            return pad_needed

        def _get_pads(auto_pad, pad_needed):
            spatial_size = len(pad_needed)
            pads = [0] * spatial_size * 2
            for i in range(spatial_size):
                if auto_pad == "SAME_LOWER":
                    pads[i] = int((pad_needed[i] + 1) / 2)
                    pads[i + spatial_size] = pad_needed[i] - pads[i]
                elif auto_pad == "SAME_UPPER":
                    pads[i] = int(pad_needed[i] / 2)
                    pads[i + spatial_size] = pad_needed[i] - pads[i]
            return pads

        input_shape = self.get_func_input_shape(n.input[0])
        strides = []
        kernel = []
        dilations = []
        pads = []
        auto_pad = "NOTSET"
        pad_mode = "constant"
        ceil_mode = 0
        value = 0.0

        if func_name == 'AveragePooling':
            func = self.generate_default_function("AveragePooling", n)
            kp = func.average_pooling_param
            pad_mode = "repeat"
        elif func_name == 'MaxPooling':
            if len(n.output) != 1:
                raise ValueError("Indices output at MaxPool is not supported")
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

        kp.ignore_border = True

        for attr in n.attribute:
            if attr.name == "strides":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for strides in {}"
                                     .format(n.op_type))
                strides.extend(attr.ints)
            elif attr.name == "pads":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for pads in {}"
                                     .format(n.op_type))
                pads.extend(attr.ints)
            elif attr.name == "kernel_shape":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for kernel_shape in {}"
                                     .format(n.op_type))
                kernel.extend(attr.ints)
            elif attr.name == "count_include_pad":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for count_include_pad in {} op_type"
                                     .format(n.op_type))
                kp.including_pad = bool(attr.i)
            elif attr.name == "dilations":
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS are supported for dilations in {}"
                                     .format(n.op_type))
                dilations.extend(attr.ints)
            elif attr.name == "auto_pad":
                if attr.type != AttributeProto.STRING:
                    raise ValueError("Only STRING is supported for auto_pad in {} op_type"
                                     .format(n.op_type))
                auto_pad = attr.s.decode("utf-8")
            elif attr.name == "ceil_mode":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for ceil_mode in {} op_type"
                                     .format(n.op_type))
                ceil_mode = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        if kernel:
            kp.kernel.dim.extend(kernel[:])
        else:
            raise ValueError("Kernel shape is required")
        if strides:
            kp.stride.dim.extend(strides[:])
        else:
            kp.stride.dim.extend([1] * len(kp.kernel.dim))
        if dilations and any(d != 1 for d in dilations):
            raise ValueError(
                "Unsupported attribute dilations was specified at {}"
                .format(n.op_type))

        if auto_pad != "NOTSET":
            if auto_pad == 'VALID':
                pads = [0] * len(kernel) * 2
                pad_needed = [0] * len(kernel)
                out_shape = _compute_output_size(
                    input_shape[2:], kp.stride.dim, kp.kernel.dim, ceil_mode, pad_needed)
            elif auto_pad in ('SAME_LOWER', 'SAME_UPPER'):
                pad_needed = _compute_pad_needed(
                    input_shape[2:], kp.stride.dim, kp.kernel.dim)
                pads = _get_pads(auto_pad, pad_needed)
                out_shape = _compute_output_size(
                    input_shape[2:], kp.stride.dim, kp.kernel.dim, ceil_mode, pad_needed)
            else:
                raise ValueError(
                    "Unsupported auto_pad type: {}".format(auto_pad))
        elif pads:
            out_shape = _compute_output_size(
                input_shape[2:], kp.stride.dim, kp.kernel.dim, ceil_mode, pads)
        else:
            out_shape = _compute_output_size(
                input_shape[2:], kp.stride.dim, kp.kernel.dim, ceil_mode, pads)

        padval = []
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
            kp.pad.dim.extend([0] * dim)
        out_shape = input_shape[:2] + out_shape
        self._shape_output[func.output[0]] = out_shape
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
                    shape.extend(list(np.ones(ndim-input0_dim, dtype=int)))
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
                    shape.extend(list(np.ones(ndim-input1_dim, dtype=int)))
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

    def Softmax_13(self, func_list, n):
        axis = -1
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

        axis = len(input_shape) + axis if axis < 0 else axis

        _reduced_shape = [1 if i == axis else input_shape[i]
                          for i in range(len(input_shape))]

        # Max
        mout_x = fork_name(n.input[0])+"_max"
        gr = generate_reduction("Max", n.name, n.input[0], mout_x,
                                [axis], True, self._graph.name, self._func_counter)
        self._shape_output[mout_x] = _reduced_shape
        func_list.append(gr)

        # Sub2
        sout_x = fork_name(n.input[0])+"_sub2"
        ga = generate_arithmetic("Sub2", n.name, [n.input[0], mout_x], sout_x,
                                 self._graph.name, self._func_counter)
        self._shape_output[sout_x] = input_shape
        func_list.append(ga)

        # Exp
        expout_x = fork_name(sout_x)+"_exp"
        ge = generate_unary("Exp", n.name, sout_x, expout_x,
                            self._graph.name, self._func_counter)
        self._shape_output[expout_x] = input_shape
        func_list.append(ge)

        # Sum
        sumout_x = fork_name(expout_x)+"_sum"
        gr = generate_reduction("Sum", n.name, expout_x, sumout_x,
                                [axis], True, self._graph.name, self._func_counter)
        self._shape_output[sumout_x] = _reduced_shape
        func_list.append(gr)

        # Div2
        ga = generate_arithmetic("Div2", n.name, [expout_x, sumout_x], n.output[0],
                                 self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = input_shape
        func_list.append(ga)

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
                # add tensor as parameter
                add_tensor_as_parameter(self._pb, t)
                self._param_vars[t.name] = None
                self._shape_output[name] = normalize_shape(t.dims)
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

    def LogSoftmax_13(self, func_list, n):
        axis = -1
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

        axis = len(input_shape) + axis if axis < 0 else axis

        _reduced_shape = [1 if i == axis else input_shape[i]
                          for i in range(len(input_shape))]

        # Max
        mout_x = fork_name(n.input[0])+"_max"
        gr = generate_reduction("Max", n.name, n.input[0], mout_x,
                                [axis], True, self._graph.name, self._func_counter)
        self._shape_output[mout_x] = _reduced_shape
        func_list.append(gr)

        # Sub2
        sout_x = fork_name(n.input[0])+"_sub2"
        ga = generate_arithmetic("Sub2", n.name, [n.input[0], mout_x], sout_x,
                                 self._graph.name, self._func_counter)
        self._shape_output[sout_x] = input_shape
        func_list.append(ga)

        # Exp
        expout_x = fork_name(sout_x)+"_exp"
        ge = generate_unary("Exp", n.name, sout_x, expout_x,
                            self._graph.name, self._func_counter)
        self._shape_output[expout_x] = input_shape
        func_list.append(ge)

        # Sum
        sumout_x = fork_name(expout_x)+"_sum"
        gr = generate_reduction("Sum", n.name, expout_x, sumout_x,
                                [axis], True, self._graph.name, self._func_counter)
        self._shape_output[sumout_x] = _reduced_shape
        func_list.append(gr)

        # Div2
        div2out_x = fork_name(n.output[0])+"_div2"
        ga = generate_arithmetic("Div2", n.name, [expout_x, sumout_x], div2out_x,
                                 self._graph.name, self._func_counter)
        self._shape_output[div2out_x] = input_shape
        func_list.append(ga)

        # Log
        ge = generate_unary("Log", n.name, div2out_x, n.output[0],
                            self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = input_shape
        func_list.append(ge)

    def Softplus(self, func_list, n):
        func = self.generate_default_function("SoftPlus", n)
        func.softplus_param.beta = 1.0
        self._shape_output[func.output[0]
                           ] = self.get_func_input_shape(func.input[0])
        func_list.append(func)

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
            del func.input[:]
            func.input.extend([n.input[0]])
            func_list.append(func)
        elif maxval is None and isinstance(minval, float):
            # Only min value is specified, so we convert to MaxScalar
            func.type = "MaximumScalar"
            msp = func.maximum_scalar_param
            msp.val = minval
            del func.input[:]
            func.input.extend([n.input[0]])
            func_list.append(func)
        elif isinstance(maxval, float) and minval is None:
            # Only max value is specified, so we use MinScalar
            func.type = "MinimumScalar"
            msp = func.minimum_scalar_param
            msp.val = maxval
            del func.input[:]
            func.input.extend([n.input[0]])
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

    def Unsqueeze(self, opset, func_list, n):
        func = self.generate_default_function("Reshape", n)
        rp = func.reshape_param
        input_shape = self.get_func_input_shape(func.input[0])
        axes = []
        if int(opset) < 13:
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
        else:
            if len(func.input) == 2:
                axes = self.get_input_raw_data(n.input[1], TensorProto.INT64)
                del func.input[1]
            else:
                raise ValueError('Onnx Unsqueeze op should has exact 2 inputs')
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

    def Squeeze(self, opset, func_list, n):
        # Convert to Reshape
        func = self.generate_default_function("Reshape", n)
        output_shape = self.get_func_input_shape(func.input[0])
        rp = func.reshape_param
        axes = []
        if int(opset) < 13:
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
        else:
            if len(func.input) == 2:
                axes_raw = self.get_input_raw_data(
                    n.input[1], TensorProto.INT64)
                axes.extend([len(output_shape) + i if i <
                             0 else i for i in axes_raw])
                del func.input[1]
            elif len(func.input) != 1:
                raise ValueError('Onnx Squeeze op should has 1 or 2 inputs')

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
        axes = [0]
        output_shape = self.get_func_input_shape(n.input[0])
        keep_dims = True
        select_last_index = False
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
                keep_dims = bool(attr.i)
            elif attr.name == "select_last_index":
                if attr.type != AttributeProto.INT:
                    raise ValueError(
                        "Only INT is supported for select_last_index in {} op_type".format(n.op_type))
                select_last_index = bool(attr.i)
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))

        # do flip when select_last_index is True
        if select_last_index:
            flip_out = fork_name(n.input[0]) + "_flip"
            flipf = self.generate_default_function('Flip', n)
            flipf.input[0] = n.input[0]
            flipf.output[0] = flip_out
            flipf.flip_param.axes.extend(axes)
            self._shape_output[flip_out] = output_shape
            func_list.append(flipf)

            block_input = flip_out
            block_output = fork_name(n.input[0]) + f'_{func_name}'
            target_axis_len = output_shape[axes[0]]
        else:
            block_input = n.input[0]
            block_output = n.output[0]

         # Convert to Max or Min
        mf = self.generate_default_function(func_name, n)
        mf.input[0] = block_input
        mf.output[0] = block_output
        if func_name == "Max":
            mp = mf.max_param
        else:
            mp = mf.min_param
        mp.only_index = True
        mp.keep_dims = keep_dims
        mp.axes.extend(axes)
        for i in mp.axes:
            if mp.keep_dims:
                output_shape[i] = 1
            else:
                del output_shape[i]
        self._shape_output[block_output] = output_shape
        func_list.append(mf)

        if select_last_index:
            # index is target_axis_len - 1 - argmax_result after Flip
            rsf = self.generate_default_function('RSubScalar', n)
            rsf.input[0] = block_output
            rsf.output[0] = n.output[0]
            rsf.r_sub_scalar_param.val = target_axis_len - 1
            self._shape_output[n.output[0]] = output_shape
            func_list.append(rsf)

    def Split(self, opset, func_list, n):
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
            elif attr.name == "split" and int(opset) < 13:
                if attr.type != AttributeProto.INTS:
                    raise ValueError(
                        "Only INTS is supported for split in {} op_type".format(n.op_type))
                output_len.extend(attr.ints)
            else:
                logger.info('Unsupported attribute {} was specified at {}'
                            .format(attr.name, n.op_type))

        if int(opset) == 13 and len(n.input) == 2:
            output_len = self.get_input_raw_data(n.input[1], TensorProto.INT64)

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
                        np.ones(ndim - len(input_shape[i]), dtype=int)) + input_shape[i]
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
                k = weight_shape[2 + index]
                i = input_shape[2 + index]
                adj = output_padding[index]
                o = 0
                if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
                    o = i * s
                    pads_value = (i - 1) * s + adj + (k - 1) * d + 1 - o
                    if auto_pad == 'SAME_LOWER':
                        pads[index + dim] = pads_value // 2
                        pads[index] = pads_value - pads[index + dim]
                    elif auto_pad == 'SAME_UPPER':
                        pads[index] = pads_value // 2
                        pads[index + dim] = pads_value - pads[index]
                else:
                    o = (i - 1) * s + adj + (k - 1) * d + 1\
                        - pads[index] - pads[index+dim]
                output_shape.append(o)

        cp.output_padding.dim.extend(output_padding)

        padval = []
        asymmetry = check_padding(pads, dim, padval)
        cp.pad.dim.extend(padval)

        if asymmetry:
            deconv_out = fork_name(func.output[0]) + "_deconv"
            del func.output[:]
            func.output.extend([deconv_out])
            deconv_out_shape = input_shape[:1] + [weight_shape[1]]\
                + [output_shape[i] + pads[i] + pads[i + dim]
                   for i in range(dim)]
            self._shape_output[deconv_out] = deconv_out_shape
            func_list.append(func)

            # Add a separate Slice Function for asymmetry padding
            start = [0, 0] + [pads[i] for i in range(dim)]
            stop = input_shape[:1] + [weight_shape[1]] + \
                [output_shape[i] + pads[i] for i in range(dim)]
            step = [1, ] * (dim + 2)
            slicef = generate_slice(n.name, deconv_out, n.output[0],
                                    start, stop, step,
                                    self._graph.name, self._func_counter)
            func_list.append(slicef)
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

    def Hardmax_6(self, func_list, n):
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

    def Hardmax_13(self, func_list, n):
        axis = -1
        input_shape = self.get_func_input_shape(n.input[0])
        for attr in n.attribute:
            if attr.name == "axis":
                axis = attr.i

        axis = len(input_shape) + axis if axis < 0 else axis
        # when axis = len(input_shape) - 1, transpose is not needed
        need_trans = len(input_shape) - 1 - axis

        if need_trans:
            # transpose target axis to last dimension
            trans0_out = fork_name(n.input[0])+"_trans0"
            i_axes = [i for i in range(len(input_shape)) if i != axis]
            i_axes.append(axis)
            trans0 = generate_transpose(n.name, n.input[0], trans0_out,
                                        i_axes, self._graph.name, self._func_counter)
            o_shape = get_transpose_output_shape(input_shape, i_axes)
            self._shape_output[trans0_out] = o_shape
            func_list.append(trans0)
            block_in = trans0_out
            block_shape = o_shape
            block_out = fork_name(n.input[0])+"_one_hot"
        else:
            block_in = n.input[0]
            block_shape = input_shape
            block_out = n.output[0]

        # max only_index
        max_func = self.generate_default_function("Max", n)
        max_out = fork_name(n.input[0])+"_max"
        max_func.input[0] = block_in
        max_func.output[0] = max_out
        mp = max_func.max_param
        mp.only_index = True
        mp.keep_dims = True
        mp.axes.extend([len(input_shape) - 1])
        max_o_shape = block_shape.copy()[:-1]
        self._shape_output[max_out] = [*max_o_shape, 1]
        func_list.append(max_func)

        # onehot
        one_hot_func = self.generate_default_function("OneHot", n)
        one_hot_func.input[0] = max_out
        one_hot_func.output[0] = block_out
        one_hot_p = one_hot_func.one_hot_param
        o_shape = block_shape.copy()
        one_hot_p.shape.dim.extend([o_shape[-1]])
        self._shape_output[block_out] = o_shape
        func_list.append(one_hot_func)

        if need_trans:
            # transpose target axis to origin dimension
            i_shape = self._shape_output[block_out].copy()
            o_axes = [*range(len(i_shape))]
            o_axes.insert(axis, o_axes.pop())
            trans1 = generate_transpose(n.name, block_out, n.output[0],
                                        o_axes, self._graph.name, self._func_counter)
            o_shape = get_transpose_output_shape(i_shape, o_axes)
            self._shape_output[n.output[0]] = o_shape
            func_list.append(trans1)

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

    def ReduceSum(self, func_list, n):
        for attr in n.attribute:
            if attr.name == 'keepdims':
                check_attr_int_type(attr, n)
                keep_dims = bool(attr.i)
            elif attr.name == 'noop_with_empty_axes':
                check_attr_int_type(attr, n)
                noop_with_empty_axes = bool(attr.i)
        if len(n.input) not in (1, 2):
            raise ValueError("Onnx ReduceSum op should has (1-2) inputs")
        if len(n.input) == 2:
            axes = self.get_input_raw_data(n.input[1], TensorProto.INT64)
        else:
            axes = []

        if not axes and noop_with_empty_axes:
            # Identity
            func = self.generate_default_function('Identity', n)
            self._shape_output[func.output[0]
                               ] = self.get_func_input_shape(func.input[0])
            func_list.append(func)
        else:
            # Sum
            func = self.generate_default_function('Sum', n)
            if len(n.input) == 2:
                del func.input[1]
            func_param = func.sum_param
            func_param.keep_dims = keep_dims
            func_param.axes.extend(axes)
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
        if len(n.input) not in [2, 3]:
            raise ValueError("Onnx QuantizeLinear op should has (2-3) inputs")
        TENSOR_TYPE_TO_INT = {
            TensorProto.INT8: 1,
            TensorProto.UINT8: 2,
        }
        input_shape = self.get_func_input_shape(n.input[0])
        axis = 1
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i

        func = self.generate_default_function("QuantizeLinear", n)

        transA_shape = [1] * len(input_shape)
        scale_shape = self.get_func_input_shape(n.input[1])
        # per axis operation when scale size > 1
        if np.prod(scale_shape) > 1:
            transA_shape[axis] = input_shape[axis]

        transA_rp_out = fork_name(n.input[1]) + "_reshape"
        rp = generate_reshape(n.name, n.input[1], transA_rp_out,
                              transA_shape, self._graph.name, self._func_counter)
        self._shape_output[transA_rp_out] = transA_shape
        func_list.append(rp)
        func.input[1] = transA_rp_out

        if len(n.input) == 3:
            transB_rp_out = fork_name(n.input[2]) + "_reshape"
            rp = generate_reshape(n.name, n.input[2], transB_rp_out,
                                  transA_shape, self._graph.name, self._func_counter)
            self._shape_output[transB_rp_out] = transA_shape
            func_list.append(rp)
            func.input[2] = transB_rp_out

        qlp = func.quantize_linear_param
        qlp.round_mode = "HALF_TO_EVEN"
        qlp.narrow_range = False

        if len(n.input) == 2:
            zero_point = fork_name(n.input[0]) + "_zero_point"
            zero_point_shape = [1] * len(input_shape)
            zero_point_array = [0]
            if np.prod(scale_shape) > 1:
                zero_point_shape[axis] = input_shape[axis]
                zero_point_array = [0] * input_shape[axis]
            create_parameter_variable(
                self._pb, zero_point, zero_point_shape, zero_point_array)
            self._param_vars[zero_point] = None
            func.input.append(zero_point)
            qlp.dtype = 2
        else:
            dtype = TensorProto.UINT8
            try:
                dtype = self.get_func_input_dtype(n.input[2])
            finally:
                qlp.dtype = TENSOR_TYPE_TO_INT[dtype]

        self._shape_output[n.output[0]] = input_shape
        func_list.append(func)

    def DequantizeLinear(self, func_list, n):
        if len(n.input) not in [2, 3]:
            raise ValueError(
                "Onnx DequantizeLinear op should has (2-3) inputs")
        input_shape = self.get_func_input_shape(n.input[0])
        axis = 1
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i
        func = self.generate_default_function("DequantizeLinear", n)

        transA_shape = [1] * len(input_shape)
        scale_shape = self.get_func_input_shape(n.input[1])
        # per axis operation when scale size > 1
        if np.prod(scale_shape) > 1:
            transA_shape[axis] = input_shape[axis]

        transA_rp_out = fork_name(n.input[1]) + "_reshape"
        rp = generate_reshape(n.name, n.input[1], transA_rp_out,
                              transA_shape, self._graph.name, self._func_counter)
        self._shape_output[transA_rp_out] = transA_shape
        func_list.append(rp)
        func.input[1] = transA_rp_out

        if len(n.input) == 3:
            transB_rp_out = fork_name(n.input[2]) + "_reshape"
            rp = generate_reshape(n.name, n.input[2], transB_rp_out,
                                  transA_shape, self._graph.name, self._func_counter)
            self._shape_output[transB_rp_out] = transA_shape
            func_list.append(rp)
            func.input[2] = transB_rp_out

        if len(n.input) == 2:
            zero_point = fork_name(n.input[0]) + "_zero_point"
            zero_point_shape = [1] * len(input_shape)
            zero_point_array = [0]
            if np.prod(scale_shape) > 1:
                zero_point_shape[axis] = input_shape[axis]
                zero_point_array = [0] * input_shape[axis]
            create_parameter_variable(
                self._pb, zero_point, zero_point_shape, zero_point_array)
            self._param_vars[zero_point] = None
            func.input.append(zero_point)

        self._shape_output[n.output[0]] = input_shape
        func_list.append(func)

    def Celu(self, func_list, n):
        # Continuously Differentiable Exponential Linear Units
        # differ from nnabla Celu(Concatenated Exponential Linear Unit)

        alpha = 1.0
        input_shape = self.get_func_input_shape(n.input[0])
        for attr in n.attribute:
            if attr.name == "alpha":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("alpha must be a single float for Op: {}"
                                     .format(n.op_type))
                alpha = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))
        # alpha
        a_cout = fork_name(n.input[0])+"_alpha"
        constant_func = self.generate_default_function("Constant", n)
        cp = constant_func.constant_param
        cp.val = alpha
        cp.shape.dim.extend(input_shape)
        del constant_func.input[0]
        constant_func.output[0] = a_cout
        func_list.append(constant_func)
        self._shape_output[a_cout] = input_shape

        # 1
        one_cout = fork_name(n.input[0])+"_one"
        constant_func1 = self.generate_default_function("Constant", n)
        cp = constant_func1.constant_param
        cp.val = 1.0
        cp.shape.dim.extend(input_shape)
        del constant_func1.input[0]
        constant_func1.output[0] = one_cout
        func_list.append(constant_func1)
        self._shape_output[one_cout] = input_shape

        # Div2
        div2out = fork_name(n.output[0])+"_div2"
        ga = generate_arithmetic("Div2", n.name, [n.input[0], a_cout], div2out,
                                 self._graph.name, self._func_counter)
        self._shape_output[div2out] = input_shape
        func_list.append(ga)

        # Exp
        expout = fork_name(div2out)+"_exp"
        ge = generate_unary("Exp", n.name, div2out, expout,
                            self._graph.name, self._func_counter)
        self._shape_output[expout] = input_shape
        func_list.append(ge)

        # Sub2
        sout = fork_name(expout)+"_sub2"
        ga = generate_arithmetic("Sub2", n.name, [expout, one_cout], sout,
                                 self._graph.name, self._func_counter)
        self._shape_output[sout] = input_shape
        func_list.append(ga)

        ms_in = sout
        if alpha != 1.0:
            # MulScalar
            muls_out = fork_name(expout)+"_muls"
            muls = generate_mul_scalar(n.name, sout, muls_out,
                                       alpha, self._graph.name, self._func_counter)
            self._shape_output[muls_out] = input_shape
            func_list.append(muls)
            ms_in = muls_out

        # MinimumScalar
        mins_out = fork_name(n.output[0])+"_mins"
        mins = generate_minimum_scalar(n.name, ms_in, mins_out,
                                       0, self._graph.name, self._func_counter)
        self._shape_output[mins_out] = input_shape
        func_list.append(mins)

        # MaximumScalar
        maxs_out = fork_name(n.output[0])+"_maxs"
        maxs = generate_maximum_scalar(n.name, n.input[0], maxs_out,
                                       0, self._graph.name, self._func_counter)
        self._shape_output[maxs_out] = input_shape
        func_list.append(maxs)

        # Add2
        ga = generate_arithmetic("Add2", n.name, [maxs_out, mins_out], n.output[0],
                                 self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = input_shape
        func_list.append(ga)

    def Shape(self, func_list, n):
        func = self.generate_default_function("Shape", n)
        input_shape = self.get_func_input_shape(n.input[0])

        start = 0
        end = len(input_shape)
        length = len(input_shape)
        for attr in n.attribute:
            if attr.name == "start":
                if attr.i < -length:
                    start = 0
                elif attr.i >= -length and attr.i < 0:
                    start = length + attr.i
                elif attr.i >= 0 and attr.i <= length:
                    start = attr.i
                elif attr.i > length:
                    start = length
            if attr.name == "end":
                if attr.i < -length:
                    end = 0
                elif attr.i >= -length and attr.i < 0:
                    end = length + attr.i
                elif attr.i >= 0 and attr.i <= length:
                    end = attr.i
                elif attr.i > length:
                    end = length

        output_shape = [1]
        sp = func.shape_param
        sp.start = start
        sp.end = end
        self._shape_output[n.output[0]] = output_shape
        func_list.append(func)

    def Resize_13(self, func_list, n):
        input_shape = self.get_func_input_shape(n.input[0])
        # ctm is abbr. for coordinate_transformation_mode
        ctm = "half_pixel"
        mode = "nearest"
        # Attributes extrapolation_value,exclude_outside,cubic_coeff_a,nearest_mode
        # are not supported
        for attr in n.attribute:
            if attr.name == "coordinate_transformation_mode":
                check_attr_string_type(attr, n)
                ctm = attr.s.decode("utf-8")
            elif attr.name == "mode":
                check_attr_string_type(attr, n)
                mode = attr.s.decode("utf-8")
        input_len = len(n.input)
        sizes = n.input[3] if input_len == 4 else None
        scales = n.input[2] if input_len >= 3 else None
        # roi input is not suppoted

        # precheck
        if sizes and scales:
            raise ValueError(
                    "Only one of 'scales' and 'sizes' can be specified in Resize.")
        if not sizes and not scales:
            raise ValueError(
                    "One of 'scales' and 'sizes' MUST be specified in Resize.")
        if ctm in ["pytorch_half_pixel", "tf_crop_and_resize"]:
            raise ValueError(
                    f"coordinate_transformation_mode {ctm} is not supported in importing Resize.")
        if mode in ["cubic"]:
            raise ValueError(
                    "mode cubic is not supported in importing Resize.")

        if sizes:
            try:
                output_size = self.get_input_raw_data(sizes, TensorProto.INT64)
            except ValueError:
                raise ValueError(
                    "sizes or scales should be prepared with initializer in importing Resize.")
        elif scales:
            try:
                scale_l = self.get_input_raw_data(scales, TensorProto.FLOAT)
            except ValueError:
                raise ValueError(
                    "sizes or scales should be prepared with initializer in importing Resize.")
            import math
            output_size = [int(math.floor(s * d))
                           for d, s in zip(input_shape, scale_l)]

        interpolate_f = self.generate_default_function("Interpolate", n)
        i_param = interpolate_f.interpolate_param
        del interpolate_f.input[:]
        interpolate_f.input.extend([n.input[0]])
        if ctm == "half_pixel":
            i_param.align_corners = False
            i_param.half_pixel = True
        elif ctm == "align_corners":
            i_param.align_corners = True
            i_param.half_pixel = False
        elif ctm == "asymmetric":
            i_param.align_corners = False
            i_param.half_pixel = False
        i_param.mode = mode
        i_param.output_size.extend(output_size)

        func_list.append(interpolate_f)
        self._shape_output[n.output[0]] = output_size

    def ScatterElements_13(self, func_list, n):
        inputs = n.input[:]
        data_shape = self.get_func_input_shape(inputs[0])
        updates_shape = self.get_func_input_shape(inputs[2])
        axis = 0
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Axis type must be a single integer")
                if attr.i < 0:
                    axis = len(data_shape) + attr.i
                else:
                    axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, n.op_type))

        # handle negative indices
        indices_shape = self.get_func_input_shape(inputs[1])
        # AddScalar
        adds_out = fork_name(inputs[1])+"_adds"
        adds = generate_add_scalar(n.name, inputs[1], adds_out,
                                   data_shape[axis], self._graph.name, self._func_counter)
        self._shape_output[adds_out] = indices_shape
        func_list.append(adds)

        # LessScalar
        lout = fork_name(inputs[1])+"_less_scalar"
        lesss = generate_less_scalar(n.name, inputs[1], lout,
                                     0, self._graph.name, self._func_counter)
        self._shape_output[lout] = indices_shape
        func_list.append(lesss)

        # Where
        where_func = self.generate_default_function("Where", n)
        del where_func.input[:]
        where_func.input.extend([lout, adds_out, inputs[1]])
        indices_out = fork_name(n.input[1])+"_where"
        del where_func.output[:]
        where_func.output.extend([indices_out])
        self._shape_output[indices_out] = indices_shape
        func_list.append(where_func)

        # handle target elements
        # scatterAdd make variance

        ones = fork_name(inputs[2]) + "_ones"
        ones_shape = updates_shape
        create_parameter_variable(self._pb, ones, ones_shape,
                                  [1] * np.prod(updates_shape))
        self._param_vars[ones] = None

        zeros = fork_name(inputs[0]) + "_zeros"
        zeros_shape = data_shape
        create_parameter_variable(self._pb, zeros, zeros_shape,
                                  [0] * np.prod(data_shape))
        self._param_vars[zeros] = None

        sa_out = fork_name(inputs[1])+"_scatter_add"
        sa_func = self.generate_default_function("ScatterAdd", n)
        sap = sa_func.scatter_add_param
        sap.axis = axis
        sa_func.input[1] = indices_out
        sa_func.input[2] = ones
        sa_func.output[0] = sa_out
        self._shape_output[sa_out] = data_shape
        func_list.append(sa_func)

        # equal
        e_out = fork_name(inputs[0])+"_equal"
        e_func = self.generate_default_function("Equal", n)
        del e_func.input[:]
        e_func.input.extend([inputs[0], sa_out])
        e_func.output[0] = e_out
        func_list.append(e_func)

        # Where
        where_func = self.generate_default_function("Where", n)
        del where_func.input[:]
        where_func.input.extend([e_out, inputs[0], zeros])
        data_out = fork_name(n.input[0])+"_where"
        del where_func.output[:]
        where_func.output.extend([data_out])
        self._shape_output[data_out] = data_shape
        func_list.append(where_func)

        sa_func = self.generate_default_function("ScatterAdd", n)
        sap = sa_func.scatter_add_param
        sap.axis = axis
        sa_func.input[0] = data_out
        sa_func.input[1] = indices_out
        self._shape_output[sa_func.output[0]] = data_shape
        func_list.append(sa_func)

    def Compress(self, func_list, n):
        input_shape = self.get_func_input_shape(n.input[0])
        con_shape = self.get_func_input_shape(n.input[1])
        if len(con_shape) != 1:
            raise ValueError(
                "onnx Compress op condition input should be rank 1.")
        axis = None
        for attr in n.attribute:
            if attr.name == "axis":
                if attr.i < 0:
                    axis = len(input_shape) + attr.i
                else:
                    axis = attr.i

        rout = n.input[0]
        indices = [*range(np.prod(con_shape))]
        if axis == None:
            # onnx flatten data when axis is none
            # reshape
            input_size = np.prod(input_shape)
            rout = fork_name(n.input[0])+"_reshape"
            rp = generate_reshape(n.name, n.input[0], rout, [input_size],
                                  self._graph.name, self._func_counter)
            self._shape_output[rout] = [input_size]
            func_list.append(rp)

        condition_out = fork_name(n.input[0])+"_indices"
        create_parameter_variable(self._pb, condition_out,
                                  con_shape, indices)
        self._param_vars[condition_out] = None

        # boolgather
        bg_out = fork_name(condition_out)+"_boolgather"
        bg = self.generate_default_function("BoolGather", n)
        del bg.input[:]
        bg.input.extend([condition_out, n.input[1]])
        bg.output[0] = bg_out
        func_list.append(bg)
        # dynamic output shape
        self._shape_output[bg_out] = con_shape

        # gather
        gf = self.generate_default_function("Gather", n)
        gparam = gf.gather_param
        gparam.axis = axis if axis else 0
        del gf.input[:]
        gf.input.extend([rout, bg_out])
        func_list.append(gf)
        # dynamic output shape
        self._shape_output[n.output[0]] = input_shape

    def Round(self, func_list, n):
        logger.warning("nnabla Round is not compatible to ONNX Round, " +
                       "which performs rounding to nearest-even integer.")
        return self.GeneralOperator('Round', func_list, n)

    def GlobalMaxPool(self, func_list, n):
        assert len(n.input) == 1
        assert len(n.output) == 1
        x_shape = self.get_func_input_shape(n.input[0])
        assert len(x_shape) >= 3  # (N, C, D0, D1, ...)

        # (N, C, D0, D1, ...) -> (N, C, 1, 1, ...)
        y_shape = [dim if i < 2 else 1 for i, dim in enumerate(x_shape)]
        axes = list(range(2, len(x_shape)))

        # Max
        max_func = self.generate_default_function("Max", n)
        max_p = max_func.max_param
        max_p.axes.extend(axes)
        max_p.keep_dims = True
        max_p.with_index = False
        max_p.only_index = False
        self._shape_output[n.output[0]] = y_shape
        func_list.append(max_func)

    def RandomNormal(self, func_list, n):
        assert len(n.input) == 0
        assert len(n.output) == 1

        dtype = int(TensorProto.FLOAT)
        mean = 0.0
        scale = 1.0
        seed = -1
        shape = None
        for attr in n.attribute:
            if attr.name == "dtype":
                dtype = attr.i
            if attr.name == "mean":
                mean = attr.f
            if attr.name == "scale":
                scale = attr.f
            if attr.name == "seed":
                attr_seed = bit_cast_f32_to_i32(attr.f)
                seed = np.iinfo(np.int32).min if attr_seed == -1 else attr_seed
            if attr.name == "shape":
                shape = normalize_shape(attr.ints)
        assert shape is not None
        if dtype not in (int(TensorProto.FLOAT), int(TensorProto.FLOAT16)):
            raise ValueError("Unsupported dtype {} was specified at {}"
                             .format(dtype, n.op_type))

        func = generate_rand_normal(n.name, n.output[0], mean, scale, seed,
                                    shape, self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = shape
        func_list.append(func)

    def RandomNormalLike(self, func_list, n):
        assert len(n.input) == 1
        assert len(n.output) == 1
        shape = self.get_func_input_shape(n.input[0])

        dtype = int(TensorProto.FLOAT)
        mean = 0.0
        scale = 1.0
        seed = -1
        for attr in n.attribute:
            if attr.name == "dtype":
                dtype = attr.i
            if attr.name == "mean":
                mean = attr.f
            if attr.name == "scale":
                scale = attr.f
            if attr.name == "seed":
                attr_seed = bit_cast_f32_to_i32(attr.f)
                seed = np.iinfo(np.int32).min if attr_seed == -1 else attr_seed
        if dtype not in (int(TensorProto.FLOAT), int(TensorProto.FLOAT16)):
            raise ValueError("Unsupported dtype {} was specified at {}"
                             .format(dtype, n.op_type))

        func = generate_rand_normal(n.name, n.output[0], mean, scale, seed,
                                    shape, self._graph.name, self._func_counter)
        self._shape_output[n.output[0]] = shape
        func_list.append(func)

    def RandomUniform(self, func_list, n):
        assert len(n.input) == 0
        assert len(n.output) == 1

        dtype = int(TensorProto.FLOAT)
        high = 1.0
        low = 0.0
        seed = -1
        shape = None
        for attr in n.attribute:
            if attr.name == "dtype":
                dtype = attr.i
            if attr.name == "high":
                high = attr.f
            if attr.name == "low":
                low = attr.f
            if attr.name == "seed":
                attr_seed = bit_cast_f32_to_i32(attr.f)
                seed = np.iinfo(np.int32).min if attr_seed == -1 else attr_seed
            if attr.name == "shape":
                shape = normalize_shape(attr.ints)
        assert shape is not None, "The shape attribute is required."
        if dtype not in (int(TensorProto.FLOAT), int(TensorProto.FLOAT16)):
            raise ValueError("Unsupported dtype {} was specified at {}"
                             .format(dtype, n.op_type))

        func = generate_rand_uniform(n.name, n.output[0], dtype, high, low,
                                     seed, shape, self._graph.name,
                                     self._func_counter)
        self._shape_output[n.output[0]] = shape
        func_list.append(func)

    def RandomUniformLike(self, func_list, n):
        assert len(n.input) == 1
        assert len(n.output) == 1
        shape = self.get_func_input_shape(n.input[0])

        dtype = int(TensorProto.FLOAT)
        high = 1.0
        low = 0.0
        seed = -1
        for attr in n.attribute:
            if attr.name == "dtype":
                dtype = attr.i
            if attr.name == "high":
                high = attr.f
            if attr.name == "low":
                low = attr.f
            if attr.name == "seed":
                attr_seed = bit_cast_f32_to_i32(attr.f)
                seed = np.iinfo(np.int32).min if attr_seed == -1 else attr_seed
        if dtype not in (int(TensorProto.FLOAT), int(TensorProto.FLOAT16)):
            raise ValueError("Unsupported dtype {} was specified at {}"
                             .format(dtype, n.op_type))

        func = generate_rand_uniform(n.name, n.output[0], dtype, high, low,
                                     seed, shape, self._graph.name,
                                     self._func_counter)
        self._shape_output[n.output[0]] = shape
        func_list.append(func)

    def TopK(self, func_list, n):
        # Get inputs
        assert len(n.input) == 2 and len(n.output) == 2
        x_shape = self.get_func_input_shape(n.input[0])
        k_data = self.get_input_raw_data(n.input[1], TensorProto.INT64)
        assert len(k_data) == 1
        k = k_data[0]

        # Get attributes
        axis = -1
        largest = 1
        sort = 1
        for attr in n.attribute:
            if attr.name == "axis":
                axis = attr.i
            elif attr.name == "largest":
                largest = attr.i
            elif attr.name == "sorted":
                sort = attr.i
            else:
                unsupported_attribute(attr.name, n)

        out0 = n.output[0]
        out1 = n.output[1]
        last_out = n.input[0]
        last_shape = x_shape

        # Transpose: (D0, ..., D_{axis}, ..., Dn) -> (D0, ..., Dn, D_{axis})
        need_transpose = (axis != -1 and axis != len(x_shape) - 1)
        if need_transpose:
            trans_out = fork_name(n.input[0]) + "_trans"
            trans_axes = list(range(len(last_shape)))
            trans_axes[axis], trans_axes[-1] = trans_axes[-1], trans_axes[axis]
            last_func = generate_transpose(n.name, last_out, trans_out,
                                           trans_axes, self._graph.name,
                                           self._func_counter)
            last_shape[axis], last_shape[-1] = last_shape[-1], last_shape[axis]
            self._shape_output[trans_out] = last_shape
            func_list.append(last_func)
            last_out = trans_out

        # TopKData: (D0, ..., Dn, D_{axis}) -> (D0, ..., Dn, k)
        # NOTICE: the TopKData documentation does not mention if its outputs
        # are sorted or not, but the CPU and CUDA implementation returns sorted
        # results.
        topk_func = self.generate_default_function("TopKData", n)
        del topk_func.input[:]
        topk_func.input.append(last_out)
        topk_p = topk_func.top_k_data_param
        topk_p.k = k
        topk_p.abs = False
        topk_p.reduce = True
        topk_p.base_axis = len(x_shape) - 1
        topk_p.largest = bool(largest)
        topk_p.with_index = True
        last_shape[-1] = k

        topk_out0 = out0
        topk_out1 = out1
        if need_transpose:
            topk_out0 = fork_name(out0) + "_top_k"
            topk_out1 = fork_name(out1) + "_top_k"
            del topk_func.output[:]
            topk_func.output.extend([topk_out0, topk_out1])
        self._shape_output[topk_out0] = last_shape
        self._shape_output[topk_out1] = last_shape
        func_list.append(topk_func)

        # Transpose: (D0, ..., Dn, k) -> (D0, ..., k, ..., Dn)
        if need_transpose:
            trans_axes = list(range(len(last_shape)))
            trans_axes[axis], trans_axes[-1] = trans_axes[-1], trans_axes[axis]
            trans_func0 = generate_transpose(n.name, topk_out0, out0,
                                             trans_axes, self._graph.name,
                                             self._func_counter)
            trans_func1 = generate_transpose(n.name, topk_out1, out1,
                                             trans_axes, self._graph.name,
                                             self._func_counter)
            last_shape[axis], last_shape[-1] = last_shape[-1], last_shape[axis]
            self._shape_output[out0] = last_shape
            self._shape_output[out1] = last_shape
            func_list.extend([trans_func0, trans_func1])

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
