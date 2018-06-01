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
import numpy as np
try:
    from onnx import (ModelProto, TensorProto, AttributeProto)
except:
    print('ONNX read support disabled.')

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
    "Reshape": "Reshape",
    "Transpose": "Transpose",
    "Abs": "Abs",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Log": "Log",
    "Less": "Less",
    "Greater": "Greater",
    "Equal": "Equal",
    "Exp": "Exp",
    "Identity": "Identity",
    # optype with different names
    "Relu": "ReLU",
    "PRelu": "PReLU",
    "Concat": "Concatenate",
    "Conv": "Convolution",
    "GlobalAveragePool": "GlobalAveragePooling",
    "MaxPool": "MaxPooling",
    "AveragePool": "AveragePooling",
    "Sum": "Add2",
    "Gemm": "Affine",
    "Add": "Add2",
    "Mul": "Mul2",
    "Div": "Div2",
    "Pow": "Pow2",
    "Sub": "Sub2",
    "MatMul": "BatchMatmul",
    "LeakyRelu": "LeakyReLU",
    "Not": "LogicalNot",
    "Elu": "ELU",
    "Selu": "SELU",
    "ReduceSum": "Sum",
    "ReduceMean": "Mean",
    "ReduceMin": "Min",
    "ReduceMax": "Max",
    "ReduceProd": "Prod",
    "And": "LogicalAnd",
    "Or": "LogicalOr",
    "Xor": "LogicalXor",
    "Max": "Maximum2",
    "Min": "Minimum2",
    "Reciprocal": "RDivScalar",
    "Neg": "MulScalar",
    "LogSoftmax": "Sub2",
    "Softplus": "Log",
    "Softsign": "Div2",
    "LRN": "Div2",
    # Clip gets converted to Identity, MaxScalar or MinScalar
    # or both, depending on the attributes.
    # We set a temporary name here
    "Clip": "Clip",
    # Constant does not get converted to a function
    # but we list it here so we can accept it
    "Constant": ""
}


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


def set_kernel_parameter_and_add_padding(node, kp, base_name, func_counter):
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
            # interleave pad values to match NNabla format
            # (S0,S1,E0,E1) => (S0,E0,S1,E1)
            half = len(pads)//2
            starts = pads[:half]
            ends = pads[half:]
            pad_width = [j for i in zip(starts, ends) for j in i]
            padf = generate_pad(node.name, input, padded,
                                "replicate", pad_width, 0,
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
                   out_name,  base_name, func_counter):
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


def convert_broadcasting_operator(func_list, node, func, base_name, func_counter):
    """Converts a broadcasting operator to a composite with BroadcastTo"""
    broadcasting = False
    broadcast_axis = -1
    for attr in node.attribute:
        if attr.name == "axis":
            if attr.type != AttributeProto.INT:
                raise ValueError("Only INT is supported for axis in {} op_type".format(node.op_type))
            broadcast_axis = attr.i
        elif attr.name == "broadcast":
            if attr.type != AttributeProto.INT:
                raise ValueError("Only INT is supported for broadcast in {} op_type".format(node.op_type))
            if attr.i == 1:
                broadcasting = True
        else:
            raise ValueError("Unsupported attribute {} was specified at {}"
                             .format(attr.name, node.op_type))
    if not broadcasting:
        return
    # Create a BroadcastTo operator to broadcast input B
    b_idx = 1  # B is the second input
    broadcasted_postfix = "_broadcasted"
    input = node.input[:]
    bin = node.input[b_idx]
    bout = bin+broadcasted_postfix
    bt = generate_broadcast_to(node.name, node.input[0], bin, bout, broadcast_axis,
                               base_name, func_counter)
    func_list.append(bt)
    input[b_idx] = bout  # rewire input to broadcasted input
    # update input with the converted inputs
    del func.input[:]
    func.input.extend(input)


def set_reduction_attrs(p, node):
    p.keep_dims = True  #  keep_dims is default True for ONNX
    for attr in node.attribute:
        if attr.name == "axes":
            if attr.type != AttributeProto.INTS:
                raise ValueError("Only INTS is supported for axes in {} op_type".format(node.op_type))
            p.axes.extend(attr.ints)
        elif attr.name == "keepdims":
            if attr.type != AttributeProto.INT:
                raise ValueError("Only INT is supported for keepdims in {} op_type".format(node.op_type))
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


def convert_to_functions(pb, network, node, base_name, initializers,
                         func_counter, param_vars, param_list, merged_inputs,
                         removed_outputs):
    """Convert given node to corresponding functions.
    A node is usually converted to a single function,
    but some nodes end up as a composition of functions,
    or a parameter (and not a function).
    """
    func_list = []
    func = generate_default_function(node, base_name, func_counter)
    if node.op_type == "Concat":
        # Concat axis was not required for Concat-1 (it is required from Concat-4),
        # so the default axis depended on which backend we use.
        # Since we are comparing with caffe2, we are
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
        func_list.append(func)
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
        func_list.append(func)
    elif node.op_type == "Dropout":
        if len(node.output) > 1:
            # An ONNX Dropout node may have two outputs (result + mak)
            # while a NNabla Dropout/Identity only allows a single output.
            # We will drop the mask output (which should be the second one).
            # This may result in a broken network (if the mask output was used later)
            # so we show a warning here
            logger.warning("Dropout's mask output {} will be removed"
                           " since NNabla does not produce mask output".format(node.output[1]))
            removed_outputs.append(node.output[1])
            del func.output[:]
            func.output.extend([node.output[0]])
        # Dropout requires a ratio to be set
        for attr in node.attribute:
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
                                 .format(attr.name, node.op_type))
        func_list.append(func)
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
            cp.stride.dim.extend(strides[:])
        if pads:
            padval = []
            asymmetry = check_padding(pads, dim, padval)
            if asymmetry:
                # Add a separate padding function for
                # asymmetry padding
                input = node.input[0]
                padded = input+"_pad"
                # interleave pad values to match NNabla format
                # (S0,S1,E0,E1) => (S0,E0,S1,E1)
                half = len(pads)//2
                starts = pads[:half]
                ends = pads[half:]
                pad_width = [j for i in zip(starts, ends) for j in i]
                padf = generate_pad(node.name, input, padded,
                                    "replicate", pad_width, 0,
                                    base_name, func_counter)
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
        func_list.append(func)
    elif node.op_type == "MaxPool":
        mpp = func.max_pooling_param
        padf = set_kernel_parameter_and_add_padding(node, mpp,
                                                    base_name, func_counter)
        if padf:
            # append a pad function if we need asymmetry padding.
            func_list.append(padf)
            # Rewire input to the padded version
            del func.input[:]
            func.input.extend(padf.output)
        # Always ignore borders in order to match ONNX(caffe2) results?
        # Not quite sure yet.
        mpp.ignore_border = True
        func_list.append(func)
    elif node.op_type == "AveragePool":
        app = func.average_pooling_param
        padf = set_kernel_parameter_and_add_padding(node, app,
                                                    base_name, func_counter)
        if padf:
            # append a pad function if we need asymmetry padding
            func_list.append(padf)
            # Rewire input to the padded version
            del func.input[:]
            func.input.extend(padf.output)
        # Always ignore borders in order to match ONNX(caffe2) results?
        # Not quite sure yet.
        app.ignore_border = True
        func_list.append(func)
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
                # BatchNormalization-1 has this field.
                # Since NNabla does not need this, we ignore it
                pass
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        func_list.append(func)
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

        # Switch H and W for transpose
        # We assume the buffer is two dimensional.
        axes = [1, 0]
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
                transA = generate_transpose(node.name, ain, aout, axes, base_name, func_counter)
                func_list.append(transA)
                input[0] = aout  # rewire input to transposed input
            elif attr.name == "transB":
                if attr.type != AttributeProto.INT:
                    raise ValueError("Only INT is supported for transB in {} op_type".format(node.op_type))
                # same as transA
                bin = node.input[1]
                bout = bin+transposed_postfix
                transB = generate_transpose(node.name, bin, bout, axes, base_name, func_counter)
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
    elif node.op_type == "Sum":
        if len(func.input) > 2:
            raise ValueError("Sum operations with more than two input is currently not supported")
        func_list.append(func)
    elif (node.op_type == "Add" or
          node.op_type == "Mul" or
          node.op_type == "Div" or
          node.op_type == "Pow" or
          node.op_type == "Sub" or
          node.op_type == "And" or
          node.op_type == "Or" or
          node.op_type == "Xor" or
          node.op_type == "Less" or
          node.op_type == "Greater" or
          node.op_type == "Equal"):
        convert_broadcasting_operator(func_list, node, func, base_name, func_counter)
        func_list.append(func)
    elif node.op_type == "Constant":
        # Convert a Constant node as an input parameter and not a function
        assert len(node.output) == 1, "Constant output must be a single buffer"
        name = node.output[0]
        for attr in node.attribute:
            if attr.name == "value":
                if attr.type != AttributeProto.TENSOR:
                    raise ValueError("Only TESNOR is supported for value in {} op_type".format(node.op_type))
                t = attr.t
                if t is None:
                    raise ValueError("value attribute must be set for {}".format(node.op_type))
                t.name = name
                # add tensor as parameter
                add_tensor_as_parameter(pb, t)
                param_vars[t.name] = None
                # Add tensor as variable
                v = network.variable.add()
                v.name = t.name
                v.shape.dim.extend(t.dims)
                v.type = "Parameter"
                v.initializer.type = "Constant"
                v.initializer.multiplier = 1.0
                param_list.append(v)
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        # We do not add any function to the list here
        # since the node is converted as a parameter
    elif node.op_type == "Reshape":
        rp = func.reshape_param
        shape_found = False
        for attr in node.attribute:
            if attr.name == "shape":
                # Shape comes as attribute for Reshape-1
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS is supported for shape in {} op_type".format(node.op_type))
                rp.shape.dim.extend(attr.ints)
                shape_found = True
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        if len(func.input) == 2:
            # Shape comes as input for Reshape-5.
            # NNabla reshape excepts a single input (data),
            # while Reshape-5 will have two inputs (data, shape),
            # so we convert the shape input to a parameter
            shape_input = func.input[1]
            # look for the initializer for matching input
            for init in initializers:
                if init.name == shape_input:
                    if init.data_type != TensorProto.INT64:
                        raise ValueError("Only INT64 is supported for shape in {} op_type".format(node.op_type))
                    # copy shape size from initializer
                    if init.raw_data:
                        rp.shape.dim.extend(np.fromstring(init.raw_data, dtype=np.int64))
                    elif init.int64_data:
                        rp.shape.dim.extend(init.int64_data)
                    shape_found = True
                    break
            # stored the merged input so we can ignore it later
            merged_inputs.append(shape_input)
            del func.input[1]
        if not shape_found:
            raise ValueError("Shape information was not found in {} op_type".format(node.op_type))
        func_list.append(func)
    elif node.op_type == "Transpose":
        tp = func.transpose_param
        for attr in node.attribute:
            if attr.name == "perm":
                # perm has the same meaning for ONNX and NNabla
                # so we simply copy the parameter
                if attr.type != AttributeProto.INTS:
                    raise ValueError("Only INTS is supported for perm in {} op_type".format(node.op_type))
                tp.axes.extend(attr.ints)
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        func_list.append(func)
    elif node.op_type == "LeakyRelu":
        lrp = func.leaky_relu_param
        lrp.alpha = 0.01  # alpha value defaults to 0.01 in ONNX
        for attr in node.attribute:
            if attr.name == "alpha":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Only FLOAT is supported for alpha in {} op_type".format(node.op_type))
                lrp.alpha = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        func_list.append(func)
    elif node.op_type == "Elu":
        ep = func.elu_param
        ep.alpha = 1.0  # alpha value defaults to 1.0 in ONNX
        for attr in node.attribute:
            if attr.name == "alpha":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Only FLOAT is supported for alpha in {} op_type".format(node.op_type))
                ep.alpha = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        func_list.append(func)
    elif node.op_type == "Selu":
        sp = func.selu_param
        sp.alpha = 1.6732  # Default value for ONNX
        sp.scale = 1.0507
        for attr in node.attribute:
            if attr.name == "alpha":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Only FLOAT is supported for alpha in {} op_type".format(node.op_type))
                sp.alpha = attr.f
            elif attr.name == "gamma":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("Only FLOAT is supported for gamma in {} op_type".format(node.op_type))
                sp.scale = attr.f
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        func_list.append(func)
    elif node.op_type == "ReduceSum":
        sp = func.sum_param
        set_reduction_attrs(sp, node)
        func_list.append(func)
    elif node.op_type == "ReduceMean":
        mp = func.mean_param
        set_reduction_attrs(mp, node)
        func_list.append(func)
    elif node.op_type == "ReduceMin":
        mp = func.min_param
        set_reduction_attrs(mp, node)
        func_list.append(func)
    elif node.op_type == "ReduceMax":
        mp = func.max_param
        set_reduction_attrs(mp, node)
        func_list.append(func)
    elif node.op_type == "ReduceProd":
        pp = func.prod_param
        set_reduction_attrs(pp, node)
        func_list.append(func)
    elif (node.op_type == "Max" or
          node.op_type == "Min"):
        # NNabla can only process two inputs for max/min.
        # Check if this is fulfilled.
        if len(node.input) != 2:
            raise ValueError("NNabla can only process Min/Max of two tensors")
        func_list.append(func)
    elif node.op_type == "PRelu":
        pp = func.prelu_param
        # ONNX PRelu defaults to the Channel axis,
        # so we set the channel axis (1) here.
        # This should be the same for NNabla
        # buf currently it defaults to 0
        # so we explicitly set 1 here.
        pp.base_axis = 1
        func_list.append(func)
    elif node.op_type == "Reciprocal":
        rp = func.r_div_scalar_param
        rp.val = 1.0
        func_list.append(func)
    elif node.op_type == "Neg":
        mp = func.mul_scalar_param
        mp.val = -1.0  # Neg is achieved by multiplying -1
        func_list.append(func)
    elif node.op_type == "LogSoftmax":
        logger.warning(SOFTMAX_WARNING)
        axis = DEFAULT_SOFTMAX_AXIS
        for attr in node.attribute:
            if attr.name == "axis":
                if attr.type != AttributeProto.INT:
                    raise ValueError("LogSoftmax axis must be a single integer")
                axis = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        # Apply Exp+Sum+Log to the input,
        # and subtract the result with the original input
        lsin = node.input[0]
        expout = lsin+"_exp"
        expf = generate_unary("Exp", node.name, lsin, expout,
                              base_name, func_counter)
        func_list.append(expf)
        sumout = expout+"_sum"
        # We keep dimension so the reduced sum can be subtracted
        # with the original input
        sumf = generate_sum(node.name, expout, sumout,
                            axis, True, base_name, func_counter)
        func_list.append(sumf)
        logout = sumout+"_log"
        log = generate_unary("Log", node.name, sumout, logout,
                             base_name, func_counter)
        func_list.append(log)
        # Rewire Sub2's input to the original input and
        # Exp+Sum+Log
        del func.input[:]
        func.input.extend([node.input[0], logout])
        func_list.append(func)
    elif node.op_type == "Clip":
        maxval = None
        minval = None
        for attr in node.attribute:
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
                                 .format(attr.name, node.op_type))
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
            minin = node.input[0]
            minout = minin+"_min"
            minf = generate_minimum_scalar(node.name, minin, minout,
                                           maxval, base_name, func_counter)
            func_list.append(minf)
            func.type = "MaximumScalar"
            del func.input[:]
            func.input.extend([minout])
            msp = func.maximum_scalar_param
            msp.val = minval
            func_list.append(func)
    elif node.op_type == "Softplus":
        # Convert to Exp+AddScalar+Log
        spin = node.input[0]
        expout = spin+"_exp"
        expf = generate_unary("Exp", node.name, spin,
                              expout, base_name, func_counter)
        func_list.append(expf)
        asout = expout+"_adds"
        asf = generate_add_scalar(node.name, expout, asout, 1.0,
                                  base_name, func_counter)
        func_list.append(asf)
        # rewire Log input to AddScalar output
        del func.input[:]
        func.input.extend([asout])
        func_list.append(func)
    elif node.op_type == "Softsign":
        # Convert to Abs+AddScalar+Div2
        ssin = node.input[0]
        expout = ssin+"_abs"
        expf = generate_unary("Abs", node.name, ssin,
                              expout, base_name, func_counter)
        func_list.append(expf)
        asout = expout+"_adds"
        asf = generate_add_scalar(node.name, expout, asout, 1.0,
                                  base_name, func_counter)
        func_list.append(asf)
        # rewire Div2 input to original input and AddScalar output
        del func.input[:]
        func.input.extend([ssin, asout])
        func_list.append(func)
    elif node.op_type == "LRN":
        # Gather attributes.
        # The following are default values for ONNX
        alpha = 1e-4
        beta = 0.75
        bias = 1.0
        size = -1
        for attr in node.attribute:
            if attr.name == "alpha":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("alpha must be a single float for Op: {}"
                                     .format(node.op_type))
                alpha = attr.f
            elif attr.name == "beta":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("beta must be a single float for Op: {}"
                                     .format(node.op_type))
                beta = attr.f
            elif attr.name == "bias":
                if attr.type != AttributeProto.FLOAT:
                    raise ValueError("bias must be a single float for Op: {}"
                                     .format(node.op_type))
                bias = attr.f
            elif attr.name == "size":
                if attr.type != AttributeProto.INT:
                    raise ValueError("size must be a single integer for Op: {}"
                                     .format(node.op_type))
                size = attr.i
            else:
                raise ValueError("Unsupported attribute {} was specified at {}"
                                 .format(attr.name, node.op_type))
        if size < 0:
            raise ValueError("Size is required for {}"
                             .format(node.op_type))
        elif (size % 2) == 0:
            raise ValueError("Only size with odd values is "
                             "currently supported for {}"
                             .format(node.op_type))

        # Convert to PowScalar+Transpose+SumPooling+Transpose+
        # MulScalar+AddScalar+PowScalar
        pow0_in = node.input[0]
        pow0_out = pow0_in+"_pow0"
        pow0 = generate_pow_scalar(node.name, pow0_in, pow0_out,
                                   2, base_name, func_counter)
        func_list.append(pow0)
        # Transpose the channel axis so we can sumpool along the channels
        # We are assuming 4D input
        trans0_out = pow0_out+"_trans0"
        trans0 = generate_transpose(node.name, pow0_out, trans0_out,
                                    [0, 2, 3, 1], base_name, func_counter)
        func_list.append(trans0)
        # SumPool along channels.
        padval = (size - 1)//2
        sp_out = trans0_out+"_sp"
        sump = generate_sum_pooling(node.name, trans0_out, sp_out,
                                    [1, size], [1, 1], True, [0, padval],
                                    base_name, func_counter)
        func_list.append(sump)
        # Transpose back
        trans1_out = sp_out+"_trans1"
        trans1 = generate_transpose(node.name, sp_out, trans1_out,
                                    [0, 3, 1, 2], base_name, func_counter)
        func_list.append(trans1)
        # MulScalar
        muls_out = trans1_out+"_muls"
        muls = generate_mul_scalar(node.name, trans1_out, muls_out,
                                   alpha/size, base_name, func_counter)
        func_list.append(muls)
        # AddScalar
        adds_out = muls_out+"_adds"
        adds = generate_add_scalar(node.name, muls_out, adds_out,
                                   bias, base_name, func_counter)
        func_list.append(adds)
        # PowScalar
        pow1_out = adds_out+"_pow1"
        pow1 = generate_pow_scalar(node.name, adds_out, pow1_out,
                                   beta, base_name, func_counter)
        func_list.append(pow1)
        # rewire Div2 input to original input and PowScalar output
        del func.input[:]
        func.input.extend([pow0_in, pow1_out])
        func_list.append(func)
    else:
        # Simply add the function for all other conversions
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

def check_domain(domain):
    # We do not allow any operator from an unknown domain
    if not (domain == '' or domain == NNABLA_DOMAIN):
        raise ValueError("Unsupported operator from domain {} was found".format(domain))


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

def onnx_graph_to_nnp_protobuf(pb, graph):
    network = pb.network.add()
    network.name = graph.name

    # We use an OrderedDict and not a set
    # to preserve order
    param_vars = OrderedDict()  # Dictionary for input parameters.
    all_vars = OrderedDict()  # Dictionary for all variables
    param_list = []  # list of parameter variables
    merged_inputs = [] # list of input buffers that was merged to a function
    removed_outputs = [] # list of output buffers that was removed
    func_counter = {}  # a counter for all functions
    # convert nodes
    for n in graph.node:
        check_domain(n.domain)
        fl = convert_to_functions(pb, network,
                                  n, graph.name, graph.initializer,
                                  func_counter, param_vars, param_list, merged_inputs,
                                  removed_outputs)
        # Gather all unique names for input and output
        for f in fl:
            for i in f.input:
                all_vars[i] = None
            for o in f.output:
                all_vars[o] = None
        network.function.extend(fl)

    # convert parameters
    for init in graph.initializer:
        if init.name in merged_inputs:
            # Ignore any initializer that is already merged
            # to a function node
            continue
        add_tensor_as_parameter(pb, init)
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
    in_list = []  # list of input variables
    out_list = []  # list of output variables
    for i in graph.input:
        if i.name in merged_inputs:
            # Ignore any input that is already merged
            # to a function node
            continue
        if i.name in param_vars:
            # This input is a parameter
            v = add_value_info_as_parameter(network, i)
            param_list.append(v)
        else:
            # This input is a buffer
            v = add_value_info_as_buffer(network, i)
            in_list.append(v)
        if i.name in all_vars:
            del all_vars[i.name]
        else:
            # We come here when a buffer (usually a parameter) is included in
            # graph.input and graph.initializer,
            # but was not actually used as input for any node.
            # No one is using this buffer so we show a warning and ignore it.
            logger.warning("Input buffer {} is not used as input for any node.".format(i.name))
    for o in graph.output:
        if o.name in removed_outputs:
            # This output buffer was removed so we are not going to 
            # use it
            continue
        v = add_value_info_as_buffer(network, o)
        out_list.append(v)
        del all_vars[v.name]

    for varg in all_vars:
        # We add all remaining variables as intermediate buffer,
        # except for the ones that was converted to a parameter.
        # A conversion of a buffer to a parameter may occur when functions
        # such as Constant is given.
        if varg in param_vars:
            continue
        # We leave the buffer size of all intermediate buffers empty
        v = network.variable.add()
        v.type = "Buffer"
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
    if model.ir_version > ONNX_IR_VERSION:
        raise ValueError("ONNX IR version newer than {} is currently not supported: {}".format(ONNX_IR_VERSION, model.ir_version))
    for opset in model.opset_import:
        if opset.domain == "":
            # ONNX opset.
            # Check if we have the correct version
            if opset.version > ONNX_OPSET_VERSION:
                raise ValueError("ONNX opset version newer than {} is currently not supported: {}".format(ONNX_OPSET_VERSION, opset.version))
        else:
            raise ValueError(
                "Unsupported opset from domain {}".format(opset.domain))

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
