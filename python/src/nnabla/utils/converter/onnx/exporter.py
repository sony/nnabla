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

import collections
import re
from functools import partial

import numpy as np
from nnabla.utils import nnabla_pb2

try:
    import onnx
    from .utils import *
    from onnx import (ModelProto, TensorProto, TensorShapeProto)
except:
    print('ONNX export support disabled because onnx python package is not found.')
    print(' You may install onnx package with "pip install onnx".')

TENSOR_TYPE_TO_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.BOOL: bool,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT8: np.int8,
    TensorProto.UINT32: np.uint32,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
}


random_seed = 0

# Helper functions


def generate_scalar_constant(output_name, tensor_name, scalar):
    """Convert a scalar value to a Constant buffer.
    This is mainly used for xxScalar operators."""
    t = onnx.helper.make_tensor(tensor_name,
                                data_type=TensorProto.FLOAT,
                                dims=[1], vals=[scalar])
    c = onnx.helper.make_node("Constant",
                              [],
                              [output_name],
                              value=t)
    return c


def generate_constant(output_name, tensor_name, data_type, dims, vals):
    t = onnx.helper.make_tensor(tensor_name,
                                data_type=data_type,
                                dims=dims, vals=vals)
    c = onnx.helper.make_node("Constant",
                              [],
                              [output_name],
                              value=t)
    return c


def add_param(graph, param_name, dtype, shape, raw_data):
    init = graph.initializer.add()
    init.name = param_name
    init.data_type = dtype
    init.dims.extend(shape)
    init.raw_data = raw_data


def generate_reshape(graph, input_name, output_name, shape):
    input_reshape_name = fork_name("reshape")
    add_param(graph, input_reshape_name, TensorProto.INT64,
              [len(shape)], shape.astype(np.int64).tostring())
    n = onnx.helper.make_node(
        'Reshape',
        [input_name, input_reshape_name],
        [output_name]
    )
    return n


def generate_pad(input_name, output_name, pad_mode, pads, value, opset):
    nl = []
    if opset == "11":
        # Constant_Pads
        pads_name = fork_name(input_name)
        c = generate_constant(pads_name, fork_name(input_name),
                              TensorProto.INT64, [len(pads)],
                              pads)
        nl.append(c)

        value_name = fork_name(input_name)
        c = generate_constant(value_name, fork_name(input_name),
                              TensorProto.FLOAT, (),
                              [value])
        nl.append(c)

        n = onnx.helper.make_node(
            "Pad",
            [input_name, pads_name, value_name],
            [output_name],
            mode=pad_mode,
        )
        nl.append(n)
    else:
        n = onnx.helper.make_node(
            "Pad",
            [input_name],
            [output_name],
            mode=pad_mode,
            pads=pads,
            value=value,
            )
        nl.append(n)

    return nl


def generate_value(type, dims, data_type, multiplier):
    d = TENSOR_TYPE_TO_DTYPE[data_type]
    if type == 'Normal':
        ret = np.random.randn(*dims) * multiplier
    elif type == 'Uniform':
        ret = np.random.uniform(-multiplier, multiplier, size=dims)
    elif type == 'Constant':
        ret = np.ones(dims) * multiplier
    else:
        raise ValueError('Generator type "' +
                         type + '" is not supported.')
    return ret.astype(d).tostring()


def get_matrix_variance(opset, graph, func_input, func_name, mean_out, axes, input_shape):
    nl = []

    # Sub
    sub_out = fork_name(func_input + "_sub")
    n = onnx.helper.make_node(
        'Sub',
        [func_input, mean_out],
        [sub_out]
    )
    nl.append(n)

    # Mul
    mul_out = fork_name(func_input + "_mul")
    n = onnx.helper.make_node(
        'Mul',
        [sub_out, sub_out],
        [mul_out]
    )
    nl.append(n)

    # ReduceSum
    sum_out = fork_name(func_input + "_sum")
    if opset == '13':
        axes_shape = (len(axes), )
        axes_param_name = fork_name("ReduceSumAxes")
        add_param(graph, axes_param_name,
                  TensorProto.INT64, axes_shape,
                  np.array(axes).astype(np.int64).tostring())
        n = onnx.helper.make_node(
            'ReduceSum',
            [mul_out, axes_param_name],
            [sum_out],
            keepdims=1,
            noop_with_empty_axes=0,
        )
    else:
        n = onnx.helper.make_node(
            'ReduceSum',
            [mul_out],
            [sum_out],
            axes=axes,
            keepdims=True
        )
    nl.append(n)

    count = [input_shape[i] for i in axes]

    # Constant
    constant = fork_name("constant")
    c = generate_constant(constant, func_name + "_constant",
                          TensorProto.FLOAT, [1],
                          [np.prod(count)])
    nl.append(c)

    # Div
    var_out = fork_name(func_input) + "_div"
    n = onnx.helper.make_node(
        'Div',
        [sum_out, constant],
        [var_out]
    )
    nl.append(n)

    return nl, var_out


def get_normalization_norm(func, mean_out, var_out, beta, gamma, constant0, constant1):
    nl = []

    # Sub, (x - x_mean)
    sub_out = fork_name(func.input[0]) + "_sub"
    n = onnx.helper.make_node("Sub",
                              [func.input[0], mean_out],
                              [sub_out]
                              )
    nl.append(n)

    # Add, (x_var + eps)
    add_out = fork_name(func.output[0]) + "_add"
    n = onnx.helper.make_node("Add",
                              [var_out, constant0],
                              [add_out])
    nl.append(n)

    # Pow, (x_var + eps) ** 0.5
    pow_out = fork_name(func.input[0]) + "_pow"
    n = onnx.helper.make_node("Pow",
                              [add_out, constant1],
                              [pow_out])
    nl.append(n)

    # Div, norm = (x - x_mean) / (x_var + eps) ** 0.5
    norm = fork_name(func.output[0]) + "_div"
    n = onnx.helper.make_node("Div",
                              [sub_out, pow_out],
                              [norm])
    nl.append(n)

    if len(func.input) == 3:
        mul_gamma_out = fork_name(func.output[0]) + "_mul"
        # Mul
        n = onnx.helper.make_node(
            "Mul",
            [norm, gamma],
            [mul_gamma_out]
        )
        nl.append(n)
        # Add
        add_beta_out = fork_name(func.output[0]) + "_add"
        n = onnx.helper.make_node(
            "Add",
            [mul_gamma_out, beta],
            [add_beta_out]
        )
        nl.append(n)
    elif len(func.input) == 2:
        if func.input[1] == 'gamma':
            # Mul
            mul_gamma_out = fork_name(func.output[0]) + "_mul"
            n = onnx.helper.make_node(
                "Mul",
                [norm, gamma],
                [mul_gamma_out]
            )
            nl.append(n)
        elif func.input[1] == 'beta':
            # Add
            add_beta_out = fork_name(func.output[0]) + "_add"
            n = onnx.helper.make_node(
                "Add",
                [norm, beta],
                [add_beta_out]
            )
            nl.append(n)

    nl[-1].output[0] = func.output[0]
    return nl


def set_reduction_attrs(node, param):
    a = onnx.helper.make_attribute("axes", param.axes)
    k = onnx.helper.make_attribute("keepdims", param.keep_dims)
    node.attribute.extend([a, k])


def create_dim(val):
    """Create a dimension message for a given dimension"""
    dim = TensorShapeProto.Dimension()
    dim.dim_value = val
    return dim


def get_tensor_type(name, type_dict):
    if name in type_dict:
        return type_dict[name]
    else:
        # Default tensor type to float
        return TensorProto.FLOAT


def replace_negative_size_with_batch_size(shape, batch_size):
    """Replace all dimensions with negative values to batch size"""
    sl = []
    for d in shape.dim:
        if d < 0:
            # Negative size means batch size
            sl.append(batch_size)
        else:
            sl.append(d)
    out_shape = nnabla_pb2.Shape()
    out_shape.dim.extend(sl)
    return out_shape


def fork_name(name):
    global random_seed
    random_seed += 1
    ret = name + '_{:04}'.format(random_seed)
    return ret


class ParameterState:
    TRANSPOSED = 1


# OnnxExporter class
class OnnxExporter:
    def __init__(self, nnp, batch_size, opset="11"):
        self._nnp = nnp.protobuf
        self._batch_size = batch_size
        self._model_proto = None
        self._net = None
        self._onehot_table = {}
        self._var_dict = {}
        self._parameters = {}
        self._parameters_state = {}
        self._input_types = {}
        self._output_types = {}
        self._broadcast_target = {}
        self._executor = None
        self._opset = 7
        self._quantize_dtype = TensorProto.INT32
        self._scale_cnt = collections.Counter()
        # Dictionary used to convert NNabla function names to ONNX op_type

        # opset_6 default op table
        table_op_set_6 = {
            "Dropout": partial(self.Dropout, "6"),
            "BatchNormalization": partial(self.BatchNormalization, "6"),
            "FusedBatchNormalization": partial(self.FusedBatchNormalization, "6"),
            "Reshape": "Reshape",
            "Transpose": "Transpose",
            "Abs": "Abs",
            "Sigmoid": "Sigmoid",
            "Tanh": "Tanh",
            "Log": "Log",
            "Less": partial(self.BinaryOperator, "Less", "6"),
            "Greater": partial(self.BinaryOperator, "Greater", "6"),
            "Equal": partial(self.BinaryOperator, "Equal", "6"),
            "Exp": "Exp",
            "Identity": "Identity",
            "Pad": partial(self.Pad, "6"),
            "ReLU": "Relu",
            "PReLU": self.PReLU,
            "LeakyReLU": "LeakyRelu",
            "Concatenate": self.Concatenate,
            "GlobalAveragePooling": "GlobalAveragePool",
            "MaxPooling": partial(self.BasePooling, 'MaxPool', "6"),
            "AveragePooling": partial(self.BasePooling, 'AveragePool', "6"),
            "Add2": partial(self.BinaryOperator, "Add", "6"),
            "BatchMatmul": self.BatchMatmul,
            "LogicalNot": "Not",
            "ELU": "Elu",
            "SELU": "Selu",
            "Sum": "ReduceSum",
            "Mean": "ReduceMean",
            "Min": "ReduceMin",
            "Max": "ReduceMax",
            "Prod": "ReduceProd",
            "Mul2": partial(self.BinaryOperator, "Mul", "6"),
            "Div2": partial(self.BinaryOperator, "Div", "6"),
            "Pow2": partial(self.BinaryOperator, "Pow", "6"),
            "Sub2": partial(self.BinaryOperator, "Sub", "6"),
            "RSubScalar": partial(self.ScalarOperator, 'RSubScalar'),
            "RDivScalar": partial(self.ScalarOperator, 'RDivScalar'),
            "RPowScalar": partial(self.ScalarOperator, 'RPowScalar'),
            "LogicalAnd": partial(self.BinaryOperator, "And", "6"),
            "LogicalOr": partial(self.BinaryOperator, "Or", "6"),
            "LogicalXor": partial(self.BinaryOperator, "Xor", "6"),
            "Maximum2": partial(self.ElementWiseCmp, "Maximum2", '6'),
            "Minimum2": partial(self.ElementWiseCmp, "Minimum2", '6'),
            "Affine": partial(self.BaseAffine, "Affine", '6'),
            "MulScalar": partial(self.ElementWiseScalar, "Mul", "6"),
            "MinimumScalar": partial(self.ElementWiseCmp, "MinimumScalar", '6'),
            "MaximumScalar": partial(self.ElementWiseCmp, "MaximumScalar", '6'),
            "AddScalar": partial(self.ElementWiseScalar, "Add", "6"),
            "PowScalar": partial(self.ElementWiseScalar, "Pow", "6"),
            "BroadcastTo": "",
            "Split": partial(self.Split, '6'),
            "Stack": partial(self.Stack, '6'),
            "Slice": partial(self.Slice, '6'),
            "Deconvolution": partial(self.BaseDeconvolution, 'Deconvolution'),
            "Flip": self.Flip,
            "OneHot": self.OneHot,
            "Unpooling": self.Unpooling_6,
            "DepthwiseConvolution": partial(self.BaseConvolution, 'DepthwiseConvolution', '6'),
            "BinaryConnectConvolution": partial(self.BaseConvolution, 'BinaryConnectConvolution', '6'),
            "Convolution": partial(self.BaseConvolution, 'Convolution', '6'),
            "BinaryConnectAffine": partial(self.BaseAffine, "BinaryConnectAffine", '6'),
            "BinaryWeightAffine": partial(self.BinaryWeightAffine, '6'),
            "BinaryWeightConvolution": partial(self.BinaryWeightConvolution, '6'),
            "Ceil": "Ceil",
            "Floor": "Floor",
            "DepthwiseDeconvolution": partial(self.BaseDeconvolution, 'DepthwiseDeconvolution'),
            "Softmax": partial(self.Softmax, '6'),
            "Embed": self.Embed,
            "Swish": self.Swish,
            "LogSoftmax": partial(self.LogSoftmax, '6'),
            "CReLU": self.CReLU,
            "ReLU6": partial(self.ReLU6, '6'),
            "HardSigmoid": "HardSigmoid",
            "HardTanh": self.HardTanh,
            "LogSigmoid": self.LogSigmoid,
            "SoftPlus": "Softplus",
            "SoftSign": "Softsign",
            "Constant": "Constant",
            "CELU": self.CELU,
            "GELU": partial(self.GELU, '6'),
            "TanhShrink": self.TanhShrink,
            "Arange": self.Arange,
            "Tile": self.Tile,
            "NotEqual": partial(self.NotEqual, '6'),
            "GreaterEqual": partial(self.GreaterEqual, '6'),
            "LessEqual": partial(self.LessEqual, '6'),
            "LogicalAndScalar": partial(self.LogicalAndScalar, '6'),
            "LogicalOrScalar": partial(self.LogicalOrScalar, '6'),
            "LogicalXorScalar": partial(self.LogicalXorScalar, '6'),
            "EqualScalar": partial(self.EqualScalar, '6'),
            "NotEqualScalar": partial(self.NotEqualScalar, '6'),
            "GreaterEqualScalar": partial(self.GreaterEqualScalar, '6'),
            "GreaterScalar": partial(self.GreaterScalar, '6'),
            "LessEqualScalar": partial(self.LessEqualScalar, '6'),
            "LessScalar": partial(self.LessScalar, '6'),
            "SpectralNorm": self.SpectralNorm,
            "WeightStandardization": partial(self.WeightStandardization, '6'),
            "LayerNormalization": partial(self.LayerNormalization, '6'),
            "InstanceNormalization": partial(self.InstanceNormalization, '6'),
            "WeightNormalization": partial(self.WeightNormalization, '6'),
            "GroupNormalization": partial(self.GroupNormalization, '6'),
        }

        table_op_set_7 = {
            "Dropout": partial(self.Dropout, "7"),
            "BatchNormalization": partial(self.BatchNormalization, "7"),
            "FusedBatchNormalization": partial(self.FusedBatchNormalization, "7"),
            "Less": partial(self.BinaryOperator, "Less", "7"),
            "Greater": partial(self.BinaryOperator, "Greater", "7"),
            "Equal": partial(self.BinaryOperator, "Equal", "7"),
            "Add2": partial(self.BinaryOperator, "Add", "7"),
            "Mul2": partial(self.BinaryOperator, "Mul", "7"),
            "Div2": partial(self.BinaryOperator, "Div", "7"),
            "Pow2": partial(self.BinaryOperator, "Pow", "7"),
            "Sub2": partial(self.BinaryOperator, "Sub", "7"),
            "LogicalAnd": partial(self.BinaryOperator, "And", "7"),
            "LogicalOr": partial(self.BinaryOperator, "Or", "7"),
            "LogicalXor": partial(self.BinaryOperator, "Xor", "7"),
            "Affine": partial(self.BaseAffine, "Affine", '7'),
            "MulScalar": partial(self.ElementWiseScalar, "Mul", "7"),
            "AddScalar": partial(self.ElementWiseScalar, "Add", "7"),
            "PowScalar": partial(self.ElementWiseScalar, "Pow", "7"),
            "Maximum2": partial(self.ElementWiseCmp, "Maximum2", '7'),
            "Minimum2": partial(self.ElementWiseCmp, "Minimum2", '7'),
            "MinimumScalar": partial(self.ElementWiseCmp, "MinimumScalar", '7'),
            "MaximumScalar": partial(self.ElementWiseCmp, "MaximumScalar", '7'),
            "SumPooling": partial(self.SumPooling, '7'),
            "Unpooling": self.Unpooling_7,
            "BinaryConnectAffine": partial(self.BaseAffine, "BinaryConnectAffine", '7'),
            "BinaryWeightAffine": partial(self.BinaryWeightAffine, '7'),
            "BinaryWeightConvolution": partial(self.BinaryWeightConvolution, '7'),
            "ACos": "Acos",
            "ASin": "Asin",
            "ATan": "Atan",
            "ATan2": self.ATan2,
            "Cos": "Cos",
            "Tan": "Tan",
            "Sin": "Sin",
            "Softmax": partial(self.Softmax, '7'),
            "GELU": partial(self.GELU, '7'),
            "LogSoftmax": partial(self.LogSoftmax, '7'),
            "NotEqual": partial(self.NotEqual, '7'),
            "GreaterEqual": partial(self.GreaterEqual, '7'),
            "LessEqual": partial(self.LessEqual, '7'),
            "LogicalAndScalar": partial(self.LogicalAndScalar, '7'),
            "LogicalOrScalar": partial(self.LogicalOrScalar, '7'),
            "LogicalXorScalar": partial(self.LogicalXorScalar, '7'),
            "EqualScalar": partial(self.EqualScalar, '7'),
            "NotEqualScalar": partial(self.NotEqualScalar, '7'),
            "GreaterEqualScalar": partial(self.GreaterEqualScalar, '7'),
            "GreaterScalar": partial(self.GreaterScalar, '7'),
            "LessEqualScalar": partial(self.LessEqualScalar, '7'),
            "LessScalar": partial(self.LessScalar, '7'),
        }
        table_op_set_7 = dict(table_op_set_6, **table_op_set_7)

        # opset_9 table
        table_op_set_9 = {
            "Unpooling": self.Unpooling_9,
            "ACosh": "Acosh",
            "ASinh": "Asinh",
            "ATanh": "Atanh",
            "Cosh": "Cosh",
            "Sign": "Sign",
            "Sinh": "Sinh",
            "BinarySigmoid": self.BinarySigmoid,
            "BinaryTanh": self.BinaryTanh,
            "Broadcast": self.Broadcast,
            "Where": "Where",
            "IsNaN": "IsNaN",
            "Sinc": self.Sinc,
            "ResetNaN": self.ResetNaN,
        }
        table_op_set_9 = dict(table_op_set_7, **table_op_set_9)

        # opset_10 table
        table_op_set_10 = {
            "IsInf": "IsInf",
            "ResetInf": self.ResetInf,
            "Slice": partial(self.Slice, '10'),
            "Unpooling": self.Unpooling_10,
            "Interpolate": self.Interpolate_10,
            "QuantizeLinear": self.QuantizeLinear,
            "DequantizeLinear": self.DequantizeLinear,
        }
        table_op_set_10 = dict(table_op_set_9, **table_op_set_10)

        # opset_11 table
        table_op_set_11 = {
            "Round": "Round",
            "Interpolate": self.Interpolate_11,
            "Pad": partial(self.Pad, "11"),
            "Unpooling": self.Unpooling_11,
            "SumPooling": partial(self.SumPooling, '11'),
            "MaxPooling": partial(self.BasePooling, 'MaxPool', '11'),
            "AveragePooling": partial(self.BasePooling, 'AveragePool', '11'),
        }
        table_op_set_11 = dict(table_op_set_10, **table_op_set_11)

        # opset_13 table
        table_op_set_13 = {
            "Softmax": partial(self.Softmax, '13'),
            "LogSoftmax": partial(self.LogSoftmax, '13'),
            "BatchNormalization": partial(self.BatchNormalization, '13'),
            "FusedBatchNormalization": partial(self.FusedBatchNormalization, '13'),
            "Sum": self.ReduceSum,
            "Split": partial(self.Split, '13'),
            "Stack": partial(self.Stack, '13'),
            "Less": partial(self.BinaryOperator, "Less", '13'),
            "Greater": partial(self.BinaryOperator, "Greater", '13'),
            "Equal": partial(self.BinaryOperator, "Equal", '13'),
            "Add2": partial(self.BinaryOperator, "Add", '13'),
            "Mul2": partial(self.BinaryOperator, "Mul", '13'),
            "Div2": partial(self.BinaryOperator, "Div", '13'),
            "Pow2": partial(self.BinaryOperator, "Pow", '13'),
            "Sub2": partial(self.BinaryOperator, "Sub", '13'),
            "LogicalAnd": partial(self.BinaryOperator, "And", '13'),
            "LogicalOr": partial(self.BinaryOperator, "Or", '13'),
            "LogicalXor": partial(self.BinaryOperator, "Xor", '13'),
            "WeightStandardization": partial(self.WeightStandardization, '13'),
            "LayerNormalization": partial(self.LayerNormalization, '13'),
            "InstanceNormalization": partial(self.InstanceNormalization, '13'),
            "WeightNormalization": partial(self.WeightNormalization, '13'),
        }
        table_op_set_13 = dict(table_op_set_11, **table_op_set_13)

        # opset_ support for SNPE
        table_op_set_snpe = {
            "Affine": partial(self.BaseAffine, "Affine", '6x'),
            "MulScalar": partial(self.ElementWiseScalar, "Mul", "6x"),
            "AddScalar": partial(self.ElementWiseScalar, "Add", "6x"),
            "Tanh": self.Tanh,
            "ReLU6": partial(self.ReLU6, "6x"),
            "MaxPooling": partial(self.BasePooling_6x, 'MaxPool', '6x'),
            "AveragePooling": partial(self.BasePooling_6x, 'AveragePool', '6x'),
            "Unpooling": self.Unpooling_9,
        }
        table_op_set_snpe = dict(table_op_set_6, **table_op_set_snpe)

        # opset_ support for TensorRT
        table_op_set_tensorrt = {
            "MaxPooling": partial(self.BasePooling_6x, 'MaxPool', '11'),
            "AveragePooling": partial(self.BasePooling_6x, 'AveragePool', '11'),
            "DequantizeLinear": self.DequantizeLinear,
            "Affine": partial(self.BaseAffine, "Affine", 'tensorrt'),
            "DepthwiseConvolution": partial(self.BaseConvolution, 'DepthwiseConvolution', 'tensorrt'),
        }
        table_op_set_tensorrt = dict(table_op_set_11, **table_op_set_tensorrt)

        opver_impl_map = {
            "6": table_op_set_6,
            "7": table_op_set_7,
            "9": table_op_set_9,
            "snpe": table_op_set_snpe,
            "10": table_op_set_10,
            "11": table_op_set_11,
            "13": table_op_set_13,
            "tensorrt": table_op_set_tensorrt
        }
        if opset.isdigit():
            opset = int(opset)
            if opset <= 6:
                self.nnabla_function_type_to_onnx_optype = opver_impl_map.get(
                    "6")
                self._opset = 6
            else:
                if str(opset) in opver_impl_map:
                    self.nnabla_function_type_to_onnx_optype = opver_impl_map.get(
                        str(opset))
                    self._opset = opset
                else:
                    self.nnabla_function_type_to_onnx_optype = table_op_set_7
                    self._opset = 7
        elif opset.lower() == 'snpe' or opset.lower() == '6x':
            self.nnabla_function_type_to_onnx_optype = table_op_set_snpe
            self._opset = 6
        elif opset.lower() == 'tensorrt':
            self.nnabla_function_type_to_onnx_optype = table_op_set_tensorrt
            self._opset = 11
        else:
            raise ValueError("Currently, No support {}".format(opset))

    def Dropout(self, opset, func):
        # Since only export executor network from nnp,
        # Dropout works only for test.
        # we set Dropout to is test mode
        n = onnx.helper.make_node(
            'Dropout',
            func.input,
            func.output,
            name=fork_name('Dropout')
        )
        if opset == "6":
            b = onnx.helper.make_attribute("is_test", 1)
            n.attribute.extend([b])
        return [n]

    def BinaryOperator(self, func_name, opset, func):
        if func_name == "And" or func_name == "Or" or func_name == "Xor":
            self._input_types[func.input[0]] = TensorProto.BOOL
            self._output_types[func.output[0]] = TensorProto.BOOL

        if func_name == "Less" or func_name == "Greater":
            self._output_types[func.output[0]] = TensorProto.BOOL

        if func_name == "Equal":
            self._output_types[func.output[0]] = TensorProto.BOOL

        # Check if the second input is a brodcast target.
        bt = func.input[1]
        nl = []
        if bt in self._broadcast_target:
            # Set the broadcast attribute to the operator
            # so we can combine BroadcastTo with this operator.
            param = self._broadcast_target[bt]
            second_input = param[0]
            axis = param[1]
            if opset == "6":
                n = onnx.helper.make_node(
                    func_name,
                    [func.input[0], second_input],
                    func.output,
                    axis=axis,
                    broadcast=1,
                    name=fork_name(func_name)
                )
                nl.append(n)
            else:  # opset >= 7
                input0_shape_len = len(self._var_dict[func.input[0]].dim)
                input1_shape_len = len(self._var_dict[second_input].dim)
                unsqueeze_output = fork_name("broadcast_unsqueeze")
                trailing = list(
                    range(input1_shape_len+1, input0_shape_len))
                axes = list(range(axis)) + trailing
                axes_shape = (len(axes), )
                if opset == '13':
                    axes_param_name = fork_name("UnsqueezeAxes")
                    add_param(self._model_proto.graph, axes_param_name,
                              TensorProto.INT64, axes_shape,
                              np.array(axes).astype(np.int64).tostring())
                    unsqueeze = onnx.helper.make_node(
                        "Unsqueeze",
                        [second_input, axes_param_name],
                        [unsqueeze_output],
                        name=fork_name("broadcast_unsqueeze"))
                else:
                    unsqueeze = onnx.helper.make_node(
                        'Unsqueeze',
                        [second_input],
                        [unsqueeze_output],
                        axes=axes,
                        name=fork_name("broadcast_unsqueeze")
                    )
                nl.append(unsqueeze)
                n = onnx.helper.make_node(
                    func_name,
                    [func.input[0], unsqueeze_output],
                    func.output,
                    name=fork_name(func_name))
                nl.append(n)
            if func_name == "And" or func_name == "Or" or func_name == "Xor":
                self._input_types[second_input] = TensorProto.BOOL
            del self._broadcast_target[bt]
        else:
            n = onnx.helper.make_node(
                func_name,
                func.input,
                func.output,
                name=fork_name(func_name))
            if opset == "6":
                b = onnx.helper.make_attribute("broadcast", 1)
                n.attribute.extend([b])
            nl.append(n)
        if func_name == "And" or func_name == "Or" or func_name == "Xor":
            self._input_types[func.input[1]] = TensorProto.BOOL
        return nl

    def BinarySigmoid(self, func):
        nl = []
        input_shape = np.array(
            [d for d in self._var_dict[func.input[0]].dim])
        c_zero_out = fork_name("constant")
        c_zero_data = np.zeros(input_shape)
        c = generate_constant(c_zero_out, func.name + "_zero",
                              TensorProto.FLOAT, input_shape,
                              c_zero_data.flatten())
        nl.append(c)

        c_one_out = fork_name("constant")
        c_one_data = np.ones(input_shape)
        c = generate_constant(c_one_out, func.name + "_one",
                              TensorProto.FLOAT, input_shape,
                              c_one_data.flatten())
        nl.append(c)

        g_out = fork_name("greater")
        n = onnx.helper.make_node(
            'Greater',
            [func.input[0], c_zero_out],
            [g_out]
        )
        nl.append(n)

        n = onnx.helper.make_node(
            'Where',
            [g_out, c_one_out, c_zero_out],
            func.output
        )
        nl.append(n)

        return nl

    def BinaryTanh(self, func):
        nl = []
        input_shape = np.array(
            [d for d in self._var_dict[func.input[0]].dim])
        c_zero_out = fork_name("constant")
        c_zero_data = np.zeros(input_shape)
        c = generate_constant(c_zero_out, func.name + "_zero",
                              TensorProto.FLOAT, input_shape,
                              c_zero_data.flatten())
        nl.append(c)

        c_one_out = fork_name("constant")
        c_one_data = np.ones(input_shape)
        c = generate_constant(c_one_out, func.name + "_one",
                              TensorProto.FLOAT, input_shape,
                              c_one_data.flatten())
        nl.append(c)

        c_neg_one_out = fork_name("constant")
        c_neg_one_data = np.full(input_shape, -1)
        c = generate_constant(c_neg_one_out, func.name + "_one",
                              TensorProto.FLOAT, input_shape,
                              c_neg_one_data.flatten())
        nl.append(c)

        g_out = fork_name("greater")
        n = onnx.helper.make_node(
            'Greater',
            [func.input[0], c_zero_out],
            [g_out]
        )
        nl.append(n)

        n = onnx.helper.make_node(
            'Where',
            [g_out, c_one_out, c_neg_one_out],
            func.output
        )
        nl.append(n)

        return nl

    def BaseConvolution(self, func_name, opset, func):
        nl = []
        input_shape = np.array(
            [d for d in self._var_dict[func.input[0]].dim])
        weight_shape = [d for d in self._var_dict[func.input[1]].dim]
        weight_base = 2
        if func_name == 'Convolution':
            cp = func.convolution_param
            inputs = func.input[:]
            group = cp.group
        elif func_name == 'BinaryConnectConvolution':
            cp = func.binary_connect_convolution_param
            inputs = [func.input[0], func.input[2]]
            if len(func.input) > 3:
                inputs += [func.input[3]]
            group = cp.group
        elif func_name == 'DepthwiseConvolution':
            cp = func.depthwise_convolution_param
            inputs = func.input[:]
            group = input_shape[cp.base_axis]
            weight_shape.insert(1, 1)
            proto_weight_shape = self._var_dict[inputs[1]]
            del proto_weight_shape.dim[:]
            proto_weight_shape.dim.extend(weight_shape)

            if opset == 'tensorrt':
                output_w_reshape_name = fork_name("output_w_reshape")
                n = generate_reshape(self._model_proto.graph, func.input[1], output_w_reshape_name,
                                     np.array(weight_shape))
                nl.append(n)
                inputs[1] = output_w_reshape_name

        else:
            raise ValueError('Internal error!')

        kernel_shape = weight_shape[weight_base:]
        dilations = cp.dilation.dim[:]
        strides = cp.stride.dim[:]
        pads = cp.pad.dim[:]
        outputs = func.output[:]
        if len(pads) == 1:  # 1-D convolution
            # Convert 1-D to 2-D for snpe
            kernel_shape += [1]
            dilations += [1]
            strides += [1]
            pads += [0]
            new_input_shape = np.array(np.concatenate(
                ([np.prod(input_shape[:cp.base_axis])], input_shape[cp.base_axis:], [1])))

            input_w_shape = np.array(
                [d for d in self._var_dict[func.input[1]].dim] + [1])
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(input_w_shape)
        elif len(pads) > 1:  # N-D convolution:
            new_input_shape = np.array(np.concatenate(
                ([np.prod(input_shape[:cp.base_axis])], input_shape[cp.base_axis:])))

        if new_input_shape.tolist() != input_shape.tolist():
            # Reshape input[0]
            output_x_reshape_name = fork_name("output_x_reshape")
            n = generate_reshape(self._model_proto.graph, func.input[0], output_x_reshape_name,
                                 new_input_shape)
            nl.append(n)
            inputs[0] = output_x_reshape_name

            # Conv
            outputs[0] = fork_name('output_conv')
            n = onnx.helper.make_node(
                'Conv',
                inputs,
                outputs,
                kernel_shape=kernel_shape,
                dilations=dilations,
                strides=strides,
                pads=pads * 2,
                group=group
            )
            nl.append(n)

            output_y_shape = np.array(
                [d for d in self._var_dict[func.output[0]].dim])
            n = generate_reshape(self._model_proto.graph, outputs[0], func.output[0],
                                 output_y_shape)
            nl.append(n)
        else:
            n = onnx.helper.make_node(
                'Conv',
                inputs,
                outputs,
                kernel_shape=kernel_shape,
                dilations=dilations,
                strides=strides,
                pads=pads * 2,
                group=group
            )
            nl.append(n)

        return nl

    def BinaryWeightConvolution(self, opset, func):
        nl = []
        cp = func.binary_weight_convolution_param
        inputs = [func.input[0], func.input[2], func.input[3]]
        if len(func.input) > 4:
            inputs += [func.input[4]]

        weight_shape = [d for d in self._var_dict[func.input[1]].dim]
        weight_base = 2

        kernel_shape = weight_shape[weight_base:]
        dilations = cp.dilation.dim[:]
        strides = cp.stride.dim[:]
        pads = cp.pad.dim[:]
        input_shape = np.array(
            [d for d in self._var_dict[func.input[0]].dim])
        output_y_shape = np.array(
            [d for d in self._var_dict[func.output[0]].dim])
        if len(pads) == 1:  # 1-D convolution
            # Convert 1-D to 2-D for snpe
            kernel_shape += [1]
            dilations += [1]
            strides += [1]
            pads += [0]
            new_input_shape = np.array(np.concatenate(
                ([np.prod(input_shape[:cp.base_axis])], input_shape[cp.base_axis:], [1])))

            # Reshape input[1]
            input_w_shape = np.array(
                [d for d in self._var_dict[func.input[1]].dim] + [1])
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(input_w_shape)
        elif len(pads) > 1:  # N-D convolution:
            new_input_shape = np.array(np.concatenate(
                ([np.prod(input_shape[:cp.base_axis])], input_shape[cp.base_axis:])))

        if new_input_shape != input_shape:
            # Reshape input[0]
            output_x_reshape_name = fork_name("output_x_reshape")
            n = generate_reshape(self._model_proto.graph, func.input[0], output_x_reshape_name,
                                 new_input_shape)
            nl.append(n)
            inputs[0] = output_x_reshape_name

        # Conv
        output_conv = fork_name('output_conv')
        conv_output_shape = np.array(np.concatenate(
                ([np.prod(output_y_shape[:cp.base_axis])], output_y_shape[cp.base_axis:])))
        n = onnx.helper.make_node(
            'Conv',
            inputs[:2],
            [output_conv],
            kernel_shape=kernel_shape,
            dilations=dilations,
            strides=strides,
            pads=pads * 2,
            group=cp.group
        )
        nl.append(n)

        input_alpha_shape = np.array(
            [d for d in self._var_dict[inputs[2]].dim])
        input_alpha_shape = np.array([input_alpha_shape[0] if input_alpha_shape[0]
                                      == conv_output_shape[i] else 1 for i in range(len(conv_output_shape))])
        proto_alpha_shape = self._var_dict[inputs[2]]
        del proto_alpha_shape.dim[:]
        proto_alpha_shape.dim.extend(input_alpha_shape)

        # Mul
        output_mul = fork_name('output_mul')
        n = onnx.helper.make_node(
            'Mul',
            [output_conv, inputs[2]],
            [output_mul],
        )
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)
        output = output_mul

        if len(inputs) > 3:
            input_bias_shape = np.array(
                [d for d in self._var_dict[inputs[3]].dim])
            input_bias_shape = np.array([input_bias_shape[0] if input_bias_shape[0]
                                         == conv_output_shape[i] else 1 for i in range(len(conv_output_shape))])
            proto_bias_shape = self._var_dict[inputs[3]]
            del proto_bias_shape.dim[:]
            proto_bias_shape.dim.extend(input_bias_shape)

            # Add
            output_add = fork_name('output_add')
            n = onnx.helper.make_node(
                'Add',
                [output_mul, inputs[3]],
                [output_add],
            )
            if opset == "6":
                b = onnx.helper.make_attribute("broadcast", 1)
                n.attribute.extend([b])
            nl.append(n)
            output = output_add

        if new_input_shape != input_shape:
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 output_y_shape)
            nl.append(n)
        else:
            nl[-1].output[0] = func.output[0]

        return nl

    def BasePooling(self, onnx_func, opset, func):
        nl = []
        input = func.input[0]
        output = func.output[0]
        input_shape = self._var_dict[input].dim
        pad_mode = "constant"
        value = 0.0

        if onnx_func == 'MaxPool':
            k = func.max_pooling_param.kernel.dim
            s = func.max_pooling_param.stride.dim
            pads = func.max_pooling_param.pad.dim
            ignore_border = func.max_pooling_param.ignore_border
            value = -np.inf
        elif onnx_func == 'AveragePool':
            k = func.average_pooling_param.kernel.dim
            s = func.average_pooling_param.stride.dim
            pads = func.average_pooling_param.pad.dim
            ignore_border = func.average_pooling_param.ignore_border
            including_pad = func.average_pooling_param.including_pad
            if not including_pad:
                pad_mode = "edge"
        else:
            raise ValueError('Internal error!')

        len_input = len(input_shape)
        len_kernel = len(k)
        diff = len_input - len_kernel
        if diff > 2:
            input_shape_reshape = np.concatenate((np.array(
                [input_shape[0], np.prod(input_shape[1:diff])]), np.array(input_shape[diff:])))
            rout = fork_name(input) + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout
            output = fork_name(func.output[0]) + "_reshape"

        pads = [d for d in pads]
        if ignore_border:
            pads = ([0, 0] + pads) * 2
        else:
            new_input_shape = [shape + pads[i]
                               for i, shape in enumerate(input_shape[-len_kernel:])]
            subs = [kk - i % ss if i % ss != 0 else kk - ss
                    for kk, ss, i in zip(k, s, new_input_shape)]
            pads = [0, 0] + pads + [0, 0] + subs

        if any(pads):
            pad_out = fork_name(input) + "_pad"
            padf = generate_pad(input, pad_out, pad_mode, pads, value, opset)
            input = pad_out
            nl.extend(padf)

        n = onnx.helper.make_node(
            onnx_func,
            [input],
            [output],
            kernel_shape=k,
            strides=s,
            pads=[0] * len_kernel * 2
        )
        nl.append(n)

        if diff > 2:
            output_shape = np.array(self._var_dict[func.output[0]].dim)
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 output_shape)
            nl.append(n)
        return nl

    def BasePooling_6x(self, onnx_func, opset, func):
        nl = []
        input = func.input[0]
        output = func.output[0]
        input_shape = self._var_dict[input].dim
        output_shape = self._var_dict[func.output[0]].dim
        opset = int(re.sub("\D", "", opset))

        if onnx_func == 'MaxPool':
            k = func.max_pooling_param.kernel.dim
            s = func.max_pooling_param.stride.dim
            pads = func.max_pooling_param.pad.dim
            ignore_border = func.max_pooling_param.ignore_border
        elif onnx_func == 'AveragePool':
            k = func.average_pooling_param.kernel.dim
            s = func.average_pooling_param.stride.dim
            pads = func.average_pooling_param.pad.dim
            ignore_border = func.average_pooling_param.ignore_border
            including_pad = func.average_pooling_param.including_pad
        else:
            raise ValueError('Internal error!')

        len_input = len(input_shape)
        len_kernel = len(k)
        diff = len_input - len_kernel
        if diff > 2:
            input_shape_reshape = np.concatenate((np.array(
                [input_shape[0], np.prod(input_shape[1:diff])]), np.array(input_shape[diff:])))
            rout = fork_name(input) + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout
            output = fork_name(func.output[0]) + "_reshape"

        pads = [d for d in pads]
        if ignore_border:
            pads = pads * 2
        else:
            new_input_shape = [shape + pads[i]
                               for i, shape in enumerate(input_shape[-len_kernel:])]
            subs = [kk - i % ss if i % ss != 0 else kk - ss
                    for kk, ss, i in zip(k, s, new_input_shape)]
            pads = pads + subs

        n = onnx.helper.make_node(
            onnx_func,
            [input],
            [output],
            kernel_shape=k,
            strides=s,
            pads=pads
        )
        if onnx_func == 'AveragePool' and opset > 6:
            c = onnx.helper.make_attribute(
                'count_include_pad', including_pad)
            n.attribute.extend([c])
        nl.append(n)

        if diff > 2:
            output_shape = np.array(self._var_dict[func.output[0]].dim)
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 output_shape)
            nl.append(n)
        return nl

    def BatchNormalization(self, opset, func, func_name="BatchNormalization"):
        nl = []
        if func_name == "BatchNormalization":
            bnp = func.batch_normalization_param
        elif func_name == "FusedBatchNormalization":
            bnp = func.fused_batch_normalization_param
        axes = bnp.axes[0]
        inputs = func.input[:]
        outputs = func.output[:]
        onnx_order = [0, 2, 1, 3, 4]
        if len(func.input) != len(onnx_order):
            raise ValueError(
                "The number of BatchNormalization input must be {}".format(len(onnx_order)))
        for p in inputs[1:]:
            d = sum([d if d > 1 else 0 for d in self._var_dict[p].dim])
            b_shape = nnabla_pb2.Shape()
            b_shape.dim.extend([d])
            self._var_dict[p] = b_shape

        input_shape = [d for d in self._var_dict[inputs[0]].dim]
        input_shape_reshape = input_shape[:]
        if axes > 1:
            input_shape_reshape = [
                np.prod(input_shape_reshape[:axes])] + input_shape_reshape[axes:]
        if len(input_shape_reshape) < 4:
            input_shape_reshape = input_shape_reshape + \
                [1] * (4 - len(input_shape_reshape))
        if input_shape_reshape != input_shape:
            output_x_reshape = fork_name("output_x_reshape")
            n = generate_reshape(self._model_proto.graph, inputs[0], output_x_reshape,
                                 np.array(input_shape_reshape))
            nl.append(n)
            inputs[0] = output_x_reshape
            outputs[0] = fork_name("reshape_output")

        if bnp.batch_stat:
            reduc_axes = list(range(len(input_shape)))
            del reduc_axes[axes]
            # ReduceMean
            mean_out = fork_name(func.input[3]) + "_reducemean"
            n = onnx.helper.make_node(
                'ReduceMean',
                [func.input[0]],
                [mean_out],
                axes=reduc_axes,
                keepdims=True
            )
            nl.append(n)

            # Squeeze
            squeeze_out = fork_name(func.input[3]) + "_squeeze"
            if opset == '13':
                reduc_axes_shape = (len(reduc_axes), )
                reduc_axes_param_name = fork_name("BnSqueezeAxes")
                add_param(self._model_proto.graph, reduc_axes_param_name,
                          TensorProto.INT64, reduc_axes_shape,
                          np.array(reduc_axes).astype(np.int64).tostring())
                n = onnx.helper.make_node(
                    'Squeeze',
                    [mean_out, reduc_axes_param_name],
                    [squeeze_out],
                )
            else:
                n = onnx.helper.make_node(
                    'Squeeze',
                    [mean_out],
                    [squeeze_out],
                    axes=reduc_axes,
                )
            nl.append(n)
            inputs[3] = squeeze_out

            # Sub
            sout = fork_name(func.input[4]) + "_sub"
            n = onnx.helper.make_node(
                'Sub',
                [func.input[0], mean_out],
                [sout],
            )
            nl.append(n)

            # Mul
            mul_out = fork_name(func.input[4]) + "_mul"
            n = onnx.helper.make_node(
                'Mul',
                [sout, sout],
                [mul_out],
            )
            nl.append(n)

            # ReduceSum
            sum_out = fork_name(func.input[4]) + "_sum"
            if opset == '13':
                reduc_axes_shape = (len(reduc_axes), )
                reduc_axes_param_name = fork_name("BnReduceSumAxes")
                add_param(self._model_proto.graph, reduc_axes_param_name,
                          TensorProto.INT64, reduc_axes_shape,
                          np.array(reduc_axes).astype(np.int64).tostring())
                n = onnx.helper.make_node(
                    'ReduceSum',
                    [mul_out, reduc_axes_param_name],
                    [sum_out],
                    keepdims=0,
                    noop_with_empty_axes=0,
                )
            else:
                n = onnx.helper.make_node(
                    'ReduceSum',
                    [mul_out],
                    [sum_out],
                    axes=reduc_axes,
                    keepdims=False
                )
            nl.append(n)

            # Constant
            constant = fork_name("constant")
            c = generate_constant(constant, func.name + "_constant",
                                  TensorProto.FLOAT, [1],
                                  [np.prod(input_shape) / input_shape[axes]])
            nl.append(c)

            # Div
            div_out = fork_name(func.input[4]) + "_div"
            n = onnx.helper.make_node(
                'Div',
                [sum_out, constant],
                [div_out],
            )
            nl.append(n)
            inputs[4] = div_out

        bn_input = [inputs[i] for i in onnx_order]
        eps = 1e-5 if bnp.eps == 0.0 else bnp.eps
        decay_rate = 0.9 if bnp.decay_rate == 0.0 \
            else bnp.decay_rate
        n = onnx.helper.make_node(
            'BatchNormalization',
            bn_input,
            [outputs[0]],
            epsilon=eps,
            momentum=decay_rate
        )
        if opset == "6":
            b = onnx.helper.make_attribute("is_test", 1)
            n.attribute.extend([b])
        nl.append(n)

        if input_shape_reshape != input_shape:
            output_y_shape = np.array(
                [d for d in self._var_dict[func.input[0]].dim])
            n = generate_reshape(self._model_proto.graph, outputs[0], func.output[0],
                                 output_y_shape)
            nl.append(n)
        return nl

    def FusedBatchNormalization(self, opset, func):
        nl = []
        bnp = func.fused_batch_normalization_param
        nonlinearity = bnp.nonlinearity
        inputs = func.input[:]
        outputs = func.output[:]

        if len(func.input) > 5:
            del func.input[5]
        bn_out = fork_name(func.input[0]) + "_bn"
        func.output[0] = bn_out
        nl.extend(self.BatchNormalization(
            opset, func, func_name="FusedBatchNormalization"))

        if len(inputs) > 5:
            # Add
            add_out = fork_name(func.input[0]) + "_add"
            n = onnx.helper.make_node(
                'Add',
                [bn_out, inputs[5]],
                [add_out],
            )
            nl.append(n)
            inputs = [add_out]
        else:
            inputs = [bn_out]

        if nonlinearity == "relu":
            # Relu
            n = onnx.helper.make_node("Relu",
                                      inputs,
                                      outputs)
            nl.append(n)
        else:
            raise ValueError(
                "Currently, nonlinearity != relu is not supported!")
        return nl

    def Concatenate(self, func):
        n = onnx.helper.make_node(
            'Concat',
            func.input,
            func.output,
            axis=func.concatenate_param.axis
        )
        return [n]

    def Unpooling_7(self, func):
        nl = []
        input = func.input[0]
        output = func.output[0]
        input_shape = np.array([d for d in self._var_dict[input].dim])
        output_shape = np.array([d for d in self._var_dict[output].dim])
        len_input_shape = len(input_shape)
        len_kernel = len(func.unpooling_param.kernel.dim)
        diff = len_input_shape - len_kernel
        if diff != 2:
            if diff < 2:
                input_shape_reshape = np.insert(input_shape, 0, 1)
            elif diff > 2:
                input_shape_reshape = np.concatenate((np.array(
                    [input_shape[0], np.prod(input_shape[1:diff])]), np.array(input_shape[diff:])))
            rout = fork_name(input) + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout
            output = fork_name(func.output[0]) + "_reshape"
        scales = list(
            map(lambda f: float(f), [1.0, 1.0] + func.unpooling_param.kernel.dim[:]))
        n = onnx.helper.make_node(
            'Upsample',
            [input],
            [output],
            name=fork_name('Upsample'),
            scales=scales
        )
        nl.append(n)
        if diff != 2:
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 output_shape)
            nl.append(n)
        return nl

    def Unpooling_9(self, func):
        nl = []
        input = func.input[0]
        output = func.output[0]
        input_shape = [d for d in self._var_dict[input].dim]
        output_shape = np.array([d for d in self._var_dict[output].dim])
        len_input_shape = len(input_shape)
        len_kernel = len(func.unpooling_param.kernel.dim)
        diff = len_input_shape - len_kernel
        if diff != 2:
            if diff < 2:
                input_shape_reshape = np.insert(input_shape, 0, 1)
            elif diff > 2:
                input_shape_reshape = np.concatenate((np.array(
                    [input_shape[0], np.prod(input_shape[1:diff])]), np.array(input_shape[diff:])))
            rout = fork_name(input) + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout
            output = fork_name(func.output[0]) + "_reshape"
        scales = np.array([1.0, 1.0] + func.unpooling_param.kernel.dim[:])
        scale_shape = (len(scales), )
        scale_param_name = fork_name("UpsampleScales")
        add_param(self._model_proto.graph, scale_param_name,
                  TensorProto.FLOAT, scale_shape,
                  scales.astype(np.float32).tostring())

        n = onnx.helper.make_node(
            'Upsample',
            [input, scale_param_name],
            [output],
            name=fork_name('Upsample')
        )
        nl.append(n)

        if diff != 2:
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 output_shape)
            nl.append(n)
        return nl

    def Unpooling_6(self, func):
        if len(func.unpooling_param.kernel.dim) != 2:
            raise ValueError("kernel.dim != 2 is not supported.")
        dims = func.unpooling_param.kernel.dim
        n = onnx.helper.make_node(
            'Upsample',
            func.input,
            func.output,
            height_scale=dims[0] * 1.0,
            width_scale=dims[1] * 1.0,
            name=fork_name('Upsample')
        )
        return [n]

    def Unpooling_10(self, func):
        input_shape = [d for d in self._var_dict[func.input[0]].dim]
        dims = list(func.unpooling_param.kernel.dim)
        dims = [1] * (len(input_shape) - len(dims)) + dims

        scale_out = fork_name(func.input[0]) + '_scale'
        add_param(self._model_proto.graph, scale_out, TensorProto.FLOAT,
                  [len(input_shape)], np.array(dims, dtype=np.float32).tostring())

        n = onnx.helper.make_node(
            "Resize",
            [func.input[0], scale_out],
            func.output,
            mode='nearest',
        )

        return [n]

    def Unpooling_11(self, func):
        input_shape = [d for d in self._var_dict[func.input[0]].dim]
        dims = list(func.unpooling_param.kernel.dim)
        dims = [1] * (len(input_shape) - len(dims)) + dims

        roi_out = fork_name(func.input[0]) + '_roi'
        roi_value = [0] * len(input_shape) + [1] * len(input_shape)
        add_param(self._model_proto.graph, roi_out,
                  TensorProto.FLOAT, [2 * len(input_shape)], np.array(roi_value, dtype=np.float32).tostring())

        scale_out = fork_name(func.input[0]) + '_scale'
        add_param(self._model_proto.graph, scale_out, TensorProto.FLOAT,
                  [len(input_shape)], np.array(dims, dtype=np.float32).tostring())

        n = onnx.helper.make_node(
            "Resize",
            [func.input[0], roi_out, scale_out],
            func.output,
            mode='nearest',
            coordinate_transformation_mode='asymmetric',
            nearest_mode='floor'
        )

        return [n]

    def OneHot(self, func):
        nl = []
        output_shape = self._var_dict[func.output[0]].dim
        if len(output_shape) != 2:
            raise ValueError('onehot dimension != 2 is not supported!')
        if output_shape[1] in self._onehot_table:
            onehot_table_name = self._onehot_table[output_shape[1]]
        else:
            onehot_table = np.zeros((output_shape[1], output_shape[1]))
            idx = np.arange(output_shape[1])
            onehot_table[idx, idx] = 1
            onehot_table_name = fork_name('onehot_table')
            raw_data = onehot_table.astype(np.float32).tostring()
            add_param(self._model_proto.graph, onehot_table_name,
                      TensorProto.FLOAT, onehot_table.shape, raw_data)
            self._onehot_table[output_shape[1]] = onehot_table_name

        flatten_output = fork_name('onehotflatten')
        n = onnx.helper.make_node(
            'Flatten',
            [func.input[0]],
            [flatten_output],
            axis=0
        )
        self._input_types[func.input[0]] = TensorProto.INT32
        nl.append(n)

        gather_out = fork_name('onehotgatherout')
        n = onnx.helper.make_node(
            'Gather',
            [onehot_table_name, flatten_output],
            [gather_out],
            axis=0
        )
        nl.append(n)

        n = generate_reshape(self._model_proto.graph, gather_out,
                             func.output[0], np.array(output_shape))
        nl.append(n)
        return nl

    def Flip(self, func):
        i = func.input[0]
        input_shape = self._var_dict[i].dim
        nl = []
        for axis in func.flip_param.axes:
            s = np.arange(len(input_shape))
            # Step 1: transpose
            perm = np.roll(s, -axis)
            o = fork_name('TransposeFlip')
            n = onnx.helper.make_node(
                'Transpose',
                [i],
                [o],
                perm=perm.tolist()
            )
            nl.append(n)

            # Step 2: gather
            index_name = fork_name('GatherFlip')
            gather_name = fork_name('GatherFlipOutput')
            raw_data = np.arange(input_shape[axis])[
                ::-1].astype(np.int32).tostring()
            index_shape = [input_shape[axis]]
            add_param(self._model_proto.graph, index_name,
                      TensorProto.INT32, index_shape, raw_data)
            n = onnx.helper.make_node(
                'Gather',
                [o, index_name],
                [gather_name],
                axis=0  # Some backend limits axis<2
            )
            nl.append(n)

            # Step 3: transpose
            perm = np.roll(s, axis)
            o = fork_name('TransposeFlip')
            n = onnx.helper.make_node(
                'Transpose',
                [gather_name],
                [o],
                perm=perm.tolist()
            )
            nl.append(n)
            i = o
        n = onnx.helper.make_node(
            'Identity',
            [o],
            func.output
        )
        nl.append(n)
        return nl

    def BaseDeconvolution(self, func_name, func):
        nl = []
        inputs = func.input[:]
        input_shape = np.array(
            [d for d in self._var_dict[func.input[0]].dim])
        weight_shape = [d for d in self._var_dict[func.input[1]].dim]
        if func_name == "Deconvolution":
            dp = func.deconvolution_param
            if dp.output_padding.dim:
                output_padding = dp.output_padding.dim[:]
            else:
                output_padding = [0] * (len(weight_shape[2:]))
            group = dp.group
        elif func_name == "DepthwiseDeconvolution":
            dp = func.depthwise_deconvolution_param
            group = input_shape[dp.base_axis] // dp.divisor
            output_padding = [0] * (len(weight_shape[1:]))
            weight_shape.insert(1, 1)
            proto_weight_shape = self._var_dict[inputs[1]]
            del proto_weight_shape.dim[:]
            proto_weight_shape.dim.extend(weight_shape)
        else:
            raise ValueError('Internal error!')

        kernel_shape = weight_shape[2:]
        dilations = dp.dilation.dim[:]
        strides = dp.stride.dim[:]
        pads = dp.pad.dim[:]
        outputs = func.output[:]
        if len(pads) == 1:  # 1-D Deconvolution
            # Convert 1-D to 2-D for snpe
            kernel_shape += [1]
            dilations += [1]
            strides += [1]
            pads += [0]
            output_padding += [0]
            new_input_shape = np.array(np.concatenate(
                ([np.prod(input_shape[:dp.base_axis])], input_shape[dp.base_axis:], [1])))

            weight_shape += [1]
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(weight_shape)
        elif len(pads) > 1:  # N-D Deconvolution
            new_input_shape = np.array(np.concatenate(
                ([np.prod(input_shape[:dp.base_axis])], input_shape[dp.base_axis:])))

        if new_input_shape.tolist() != input_shape.tolist():
            # Reshape input[0]
            output_x_reshape = fork_name("output_x_reshape")
            n = generate_reshape(self._model_proto.graph, func.input[0],
                                 output_x_reshape, new_input_shape)
            nl.append(n)
            inputs[0] = output_x_reshape

            # ConvTranspose
            outputs[0] = fork_name('output_convtranspose')
            n = onnx.helper.make_node(
                'ConvTranspose',
                inputs,
                outputs,
                kernel_shape=kernel_shape,
                dilations=dilations,
                strides=strides,
                pads=pads * 2,
                group=group,
                output_padding=output_padding
            )
            nl.append(n)

            output_y_shape = np.array(
                [d for d in self._var_dict[func.output[0]].dim])
            n = generate_reshape(self._model_proto.graph, outputs[0],
                                 func.output[0], output_y_shape)
            nl.append(n)
        else:
            n = onnx.helper.make_node(
                'ConvTranspose',
                inputs,
                outputs,
                kernel_shape=kernel_shape,
                dilations=dilations,
                strides=strides,
                pads=pads * 2,
                group=group,
                output_padding=output_padding
            )
            nl.append(n)

        return nl

    def _elem_op(self, func, op_name, val):
        nl = []
        x = func.input[0]
        sval = fork_name(x) + "_scalar"
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        val = [val] * np.prod(input_shape)
        c = generate_constant(sval, func.name + "_scalar",
                              TensorProto.FLOAT, input_shape, val)
        nl.append(c)
        n = onnx.helper.make_node(
            op_name,
            [sval, x],
            func.output,
            name=fork_name(op_name)
        )
        nl.append(n)
        return nl

    def ScalarOperator(self, nn_funcname, func):
        if nn_funcname == 'RSubScalar':
            val = func.r_sub_scalar_param.val
            return self._elem_op(func, 'Sub', val)
        elif nn_funcname == 'RDivScalar':
            val = func.r_div_scalar_param.val
            return self._elem_op(func, 'Div', val)
        elif nn_funcname == 'RPowScalar':
            val = func.r_pow_scalar_param.val
            return self._elem_op(func, 'Pow', val)
        return []

    def Slice(self, opset, func):
        nl = []
        starts = [d for d in func.slice_param.start]
        stops = [d for d in func.slice_param.stop]
        step = [d for d in func.slice_param.step]
        input_shape = self._var_dict[func.input[0]].dim
        axes = list(range(len(input_shape) - len(starts), len(input_shape)))
        if opset == '6':
            for i in step:
                if i != 1:
                    raise ValueError('Currently, step != 1 not supported!')
            n = onnx.helper.make_node(
                "Slice",
                func.input,
                func.output,
                starts=starts,
                ends=stops,
                axes=axes
            )
            nl.append(n)
        else:
            inputs = func.input
            starts_out = fork_name(func.input[0]) + "_start"
            starts_c = generate_constant(starts_out, func.name + "_starts",
                                         TensorProto.INT64, [len(starts)], starts)
            nl.append(starts_c)
            inputs.append(starts_out)
            stops_out = fork_name(func.input[0]) + "_stop"
            stops_c = generate_constant(stops_out, func.name + "_stops",
                                        TensorProto.INT64, [len(stops)], stops)
            nl.append(stops_c)
            inputs.append(stops_out)
            axes_out = fork_name(func.input[0]) + "_axes"
            axes_c = generate_constant(axes_out, func.name + "_axes",
                                       TensorProto.INT64, [len(axes)], axes)
            nl.append(axes_c)
            inputs.append(axes_out)
            step_out = fork_name(func.input[0]) + "_step"
            step_c = generate_constant(step_out, func.name + "_step",
                                       TensorProto.INT64, [len(step)], step)
            nl.append(step_c)
            inputs.append(step_out)
            n = onnx.helper.make_node(
                "Slice",
                inputs,
                func.output,
            )
            nl.append(n)
        return nl

    def Stack(self, opset, func):
        nl = []
        outputs = []
        if opset == '13':
            axes = [func.stack_param.axis]
            axes_shape = (len(axes), )
            for i, x in enumerate(func.input):
                axes_param_name = fork_name("StackAxes")
                add_param(self._model_proto.graph, axes_param_name,
                          TensorProto.INT64, axes_shape,
                          np.array(axes).astype(np.int64).tostring())
                output_name = fork_name(x)
                n = onnx.helper.make_node(
                    "Unsqueeze",
                    [x, axes_param_name],
                    [output_name],
                    name=fork_name("Unsqueeze"))
                nl.append(n)
                outputs.append(output_name)
        else:
            for i, x in enumerate(func.input):
                output_name = fork_name(x)
                n = onnx.helper.make_node(
                    "Unsqueeze",
                    [x],
                    [output_name],
                    name=fork_name("Unsqueeze"))
                attr = onnx.helper.make_attribute(
                    "axes", [func.stack_param.axis])
                n.attribute.extend([attr])
                nl.append(n)
                outputs.append(output_name)
        n = onnx.helper.make_node(
            "Concat",
            outputs,
            func.output,
            name=fork_name("Concat"))
        attr = onnx.helper.make_attribute("axis", func.stack_param.axis)
        n.attribute.extend([attr])
        nl.append(n)
        return nl

    def Split(self, opset, func):
        nl = []
        outputs = [fork_name(out) for out in func.output]

        n = onnx.helper.make_node(
            "Split",
            func.input,
            outputs,
            name=fork_name("Split"))
        attr = onnx.helper.make_attribute("axis", func.split_param.axis)
        n.attribute.extend([attr])
        nl.append(n)

        if opset == '13':
            axes = [func.split_param.axis]
            axes_shape = (len(axes), )
            for i, x in enumerate(outputs):
                axes_param_name = fork_name("SplitAxes")
                add_param(self._model_proto.graph, axes_param_name,
                          TensorProto.INT64, axes_shape,
                          np.array(axes).astype(np.int64).tostring())

                n = onnx.helper.make_node(
                    "Squeeze",
                    [x, axes_param_name],
                    [func.output[i]],
                    name=fork_name("Squeeze"))
                nl.append(n)
        else:
            for i, x in enumerate(outputs):
                n = onnx.helper.make_node(
                    "Squeeze",
                    [x],
                    [func.output[i]],
                    name=fork_name("Squeeze"))
                attr = onnx.helper.make_attribute(
                    "axes", [func.split_param.axis])
                n.attribute.extend([attr])
                nl.append(n)
        return nl

    def BaseAffine(self, func_name, opset, func):
        """
        Affine is decomposed as 3 steps:
            Reshape inputs
            Gemm
            Reshape
        """
        nl = []
        if func_name == 'Affine':
            ap = func.affine_param
            inputs = func.input[:]
        elif func_name == 'BinaryConnectAffine':
            ap = func.binary_connect_affine_param
            inputs = [func.input[0], func.input[2]]
            if len(func.input) > 3:
                inputs += [func.input[3]]
        else:
            raise ValueError('Internal error!')

        base_axis = ap.base_axis
        x_shape = list(self._var_dict[inputs[0]].dim[:])
        w_shape = list(self._var_dict[inputs[1]].dim[:])
        y_shape = list(self._var_dict[func.output[0]].dim[:])
        x_shape_dims = [np.prod(x_shape[:base_axis]),
                        np.prod(x_shape[base_axis:])]
        gemm_output_shape = [np.prod(x_shape[:base_axis]),
                             np.prod(w_shape[1:])]

        if x_shape_dims != x_shape:
            rout = fork_name(inputs[0]) + "_reshape"
            n = generate_reshape(self._model_proto.graph, inputs[0],
                                 rout, np.array(x_shape_dims))
            nl.append(n)
            inputs[0] = rout

        # For quantized weight
        if opset != 'tensorrt':
            # To support SNPE, default set to `transB=1`
            if func.input[1] not in self._parameters:
                raise ValueError(
                    "{} is not in network's parameters.".format(func.input[1]))

        transB = 0
        if opset == '6x':
            state = self._parameters_state.get(inputs[1], 0)
            if not state & ParameterState.TRANSPOSED:
                # make it to `transB=1`
                w_shape_dims = [int(np.prod(w_shape) / w_shape[0]), w_shape[0]]
                proto_w_shape = self._var_dict[inputs[1]]
                del proto_w_shape.dim[:]
                proto_w_shape.dim.extend(w_shape_dims)
                param = self._parameters[inputs[1]]
                w_data = np.array(param.data).reshape(
                    w_shape[0], int(np.prod(w_shape) / w_shape[0]))
                d = list(np.transpose(w_data).astype(np.float32).flatten())
                del param.data[:]
                param.data.extend(d)
                self._parameters_state[inputs[1]
                                       ] = state | ParameterState.TRANSPOSED
            transB = 1
        else:
            w_shape_dims = [w_shape[0], int(np.prod(w_shape) / w_shape[0])]
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(w_shape_dims)

        if len(inputs) <= 2:
            inputs.append(fork_name("affine_bias"))
            shape = [gemm_output_shape[1]]
            raw_data = np.zeros(shape).astype(np.float32).tostring()
            add_param(self._model_proto.graph,
                      inputs[2], TensorProto.FLOAT, shape, raw_data)
        else:
            bias_shape = list(self._var_dict[inputs[2]].dim[:])
            new_bias_shape = [np.prod(bias_shape)]
            proto_bias_shape = nnabla_pb2.Shape()
            proto_bias_shape.dim.extend(new_bias_shape)
            self._var_dict[inputs[2]] = proto_bias_shape

        if gemm_output_shape == y_shape:
            n = onnx.helper.make_node(
                "Gemm",
                inputs,
                func.output,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=transB,
                name=fork_name("Gemm"))
            if opset == "6" or opset == "6x":
                b = onnx.helper.make_attribute("broadcast", 1)
                n.attribute.extend([b])
            nl.append(n)
        else:
            gemm_out = fork_name(func.output[0])
            n = onnx.helper.make_node(
                "Gemm",
                inputs,
                [gemm_out],
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=transB,
                name=fork_name("Gemm"))
            if opset == "6" or opset == "6x":
                b = onnx.helper.make_attribute("broadcast", 1)
                n.attribute.extend([b])
            nl.append(n)

            n = generate_reshape(self._model_proto.graph, gemm_out,
                                 func.output[0], np.array(y_shape))
            nl.append(n)

        return nl

    def BinaryWeightAffine(self, opset, func):
        nl = []
        bp = func.binary_weight_affine_param
        inputs = [func.input[0], func.input[2], func.input[3]]
        if len(func.input) > 4:
            inputs += [func.input[4]]
        output = func.output[0]
        out_a = fork_name(inputs[0])
        base_axis = bp.base_axis

        x_shape = list(self._var_dict[inputs[0]].dim[:])
        x_shape_dims = [np.prod(x_shape[:base_axis]),
                        np.prod(x_shape[base_axis:])]
        x_shape_dims_name = fork_name('x_shape_dims')
        x_shape_dims_raw = np.array(x_shape_dims).astype(np.int64)
        add_param(self._model_proto.graph, x_shape_dims_name, TensorProto.INT64, list(
            x_shape_dims_raw.shape), x_shape_dims_raw.tostring())

        n = onnx.helper.make_node(
            "Reshape",
            [inputs[0], x_shape_dims_name],
            [out_a]
        )
        nl.append(n)
        inputs[0] = out_a

        w_shape = list(self._var_dict[inputs[1]].dim[:])
        w_shape_dims = [w_shape[0], int(np.prod(w_shape) / w_shape[0])]
        proto_w_shape = self._var_dict[inputs[1]]
        del proto_w_shape.dim[:]
        proto_w_shape.dim.extend(w_shape_dims)

        # MatMul
        matmul_out = fork_name("matmul")
        n = onnx.helper.make_node(
            "MatMul",
            inputs[:2],
            [matmul_out]
        )
        nl.append(n)

        # Mul
        mul_out = fork_name("mul")
        n = onnx.helper.make_node(
            "Mul",
            [matmul_out, inputs[2]],
            [mul_out]
        )
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)
        output = mul_out

        # Add
        if len(inputs) > 3:
            bias_shape = list(self._var_dict[inputs[3]].dim[:])
            new_bias_shape = [np.prod(bias_shape)]
            proto_bias_shape = nnabla_pb2.Shape()
            proto_bias_shape.dim.extend(new_bias_shape)
            self._var_dict[inputs[3]] = proto_bias_shape

            add_out = fork_name(func.output[0])
            n = onnx.helper.make_node(
                "Add",
                [mul_out, inputs[3]],
                [add_out]
            )
            if opset == "6":
                b = onnx.helper.make_attribute("broadcast", 1)
                n.attribute.extend([b])
            nl.append(n)
            output = add_out

        # Reshape
        output_shape = np.array(
            self._var_dict[func.output[0]].dim)
        n = generate_reshape(self._model_proto.graph, output,
                             func.output[0], output_shape)
        nl.append(n)

        return nl

    def SumPooling(self, opset, func):
        # SumPooling gets converted to AveragePooling+Mul.
        # Mul is used to counter the division in AveragePooling
        # since SumPooling is just summing the values in each kernel.
        # Copy kernel, stride, and pads values
        nl = []
        spp = func.sum_pooling_param
        input = func.input[0]
        input_shape = list(self._var_dict[input].dim[:])
        k = spp.kernel.dim
        s = spp.stride.dim
        pads = spp.pad.dim
        ignore_border = spp.ignore_border

        len_input = len(input_shape)
        len_kernel = len(k)
        diff = len_input - len_kernel
        if diff > 2:
            input_shape_reshape = np.concatenate((np.array(
                [input_shape[0], np.prod(input_shape[1:diff])]), np.array(input_shape[diff:])))
            rout = fork_name(input) + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout

        pads = [d for d in pads]
        if ignore_border:
            pads = ([0, 0] + pads) * 2
        else:
            new_input_shape = [shape + pads[i]
                               for i, shape in enumerate(input_shape[-len_kernel:])]
            subs = [kk - i % ss if i % ss != 0 else kk - ss
                    for kk, ss, i in zip(k, s, new_input_shape)]
            pads = [0, 0] + pads + [0, 0] + subs

        if any(pads):
            pad_out = fork_name(input) + "_pad"
            padf = generate_pad(input, pad_out, 'constant', pads, 0.0, opset)
            input = pad_out
            nl.extend(padf)

        apout = fork_name(input) + "_ap"
        n = onnx.helper.make_node(
            "AveragePool",
            [input],
            [apout],
            kernel_shape=k,
            strides=s,
            pads=[0] * len_kernel * 2,
            count_include_pad=1
        )
        nl.append(n)

        rout = fork_name(func.output[0]) + "_reshape"
        output_shape = np.array(self._var_dict[func.output[0]].dim)
        n = generate_reshape(self._model_proto.graph, apout, rout,
                             output_shape)
        nl.append(n)

        # Counter the averaging process by multiplying kernel size
        kernel_size = np.prod(spp.kernel.dim)
        mulout = fork_name(input) + "_kernel"
        c = generate_scalar_constant(
            mulout, func.name + "_kernel", kernel_size)
        nl.append(c)
        n = onnx.helper.make_node("Mul",
                                  [rout, mulout],
                                  func.output)
        nl.append(n)
        return nl

    def ElementWiseScalar(self, func_name, opset, func):
        nl = []
        if func_name == "Mul":
            sp = func.mul_scalar_param
            if sp.val == -1.0:
                n = onnx.helper.make_node("Neg",
                                          func.input,
                                          func.output)
                nl.append(n)
                return nl
        elif func_name == "Add":
            sp = func.add_scalar_param
        elif func_name == "Pow":
            sp = func.pow_scalar_param
        else:
            raise ValueError(
                "{} is not support".format(func_name))
        # Convert the scalar param to a Const node and add it with input
        x = func.input[0]
        sval = fork_name(x) + "_scalar"
        if opset == "6x":  # To support SNPE
            input_shape = list(self._var_dict[func.input[0]].dim[:])
            val = [sp.val] * np.prod(input_shape)
            c = generate_constant(sval, func.name + "_scalar",
                                  TensorProto.FLOAT, input_shape, val)
        else:
            c = generate_scalar_constant(sval, func.name + "_scalar", sp.val)
        nl.append(c)
        n = onnx.helper.make_node(func_name,
                                  [x, sval],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)
        return nl

    def ElementWiseCmp(self, func_name, opset, func):
        nl = []
        output_shape = []
        inputs = func.input[:]
        input_shape0 = list(self._var_dict[inputs[0]].dim[:])
        if func_name == "Maximum2":
            onnx_func = "Max"
        elif func_name == "Minimum2":
            onnx_func = "Min"
        elif func_name == "MinimumScalar":
            onnx_func = "Min"
            msp = func.minimum_scalar_param
            sval = fork_name("scalar_value")
            c = generate_scalar_constant(sval, func.name + "_scalar", msp.val)
            nl.append(c)
            shape = nnabla_pb2.Shape()
            shape.dim.extend([1])
            self._var_dict[sval] = shape
            output_shape = input_shape0
            inputs.append(sval)
        elif func_name == "MaximumScalar":
            onnx_func = "Max"
            msp = func.maximum_scalar_param
            sval = fork_name("scalar_value")
            c = generate_scalar_constant(sval, func.name + "_scalar", msp.val)
            nl.append(c)
            shape = nnabla_pb2.Shape()
            shape.dim.extend([1])
            self._var_dict[sval] = shape
            output_shape = input_shape0
            inputs.append(sval)
        else:
            raise ValueError(
                "{} is not support".format(func_name))

        if len(output_shape) == 0:
            input_shape1 = list(self._var_dict[inputs[1]].dim[:])
            output_shape = [input_shape0[i] if input_shape0[i] !=
                            1 else input_shape1[i] for i in range(len(input_shape0))]

        for i in range(2):
            shape = list(self._var_dict[inputs[i]].dim[:])
            if shape != output_shape:
                c_zero_out = fork_name("constant")
                c_zero_data = np.zeros(output_shape)
                c = generate_constant(c_zero_out, func.name + "_zero",
                                      TensorProto.FLOAT, output_shape,
                                      c_zero_data.flatten())
                nl.append(c)

                aout = fork_name("Add")
                n = onnx.helper.make_node(
                    "Add",
                    [c_zero_out, inputs[i]],
                    [aout]
                )
                if opset == "6":
                    b = onnx.helper.make_attribute("broadcast", 1)
                    n.attribute.extend([b])
                nl.append(n)
                inputs[i] = aout

        n = onnx.helper.make_node(
            onnx_func,
            inputs,
            func.output,
        )
        shape = nnabla_pb2.Shape()
        shape.dim.extend(output_shape)
        self._var_dict[func.output[0]] = shape
        nl.append(n)

        return nl

    def ATan2(self, func):
        # Convert Div+Atan
        nl = []
        # Div
        dout = fork_name(func.output[0]) + "_div"
        n = onnx.helper.make_node("Div",
                                  func.input,
                                  [dout])
        nl.append(n)

        # Atan
        n = onnx.helper.make_node("Atan",
                                  [dout],
                                  func.output)
        nl.append(n)
        return nl

    def BatchMatmul(self, func):
        # Convert Transpose+Reshape+MatMul
        nl = []
        bmp = func.batch_matmul_param
        inputs = func.input[:]
        input_a_shape = list(self._var_dict[inputs[0]].dim[:])
        input_b_shape = list(self._var_dict[inputs[1]].dim[:])

        # Transpse input_a
        if bmp.transpose_a:
            transpose_out = fork_name(inputs[0]) + "_transpose"
            transpose = list(range(len(input_a_shape)))
            transpose[-1], transpose[-2] = transpose[-2], transpose[-1]
            n = onnx.helper.make_node(
                'Transpose',
                [inputs[0]],
                [transpose_out],
                perm=transpose
            )
            nl.append(n)
            inputs[0] = transpose_out

        # Transpse input_b
        if bmp.transpose_b:
            transpose_out = fork_name(inputs[1]) + "_transpose"
            transpose = list(range(len(input_a_shape)))
            transpose[-1], transpose[-2] = transpose[-2], transpose[-1]
            n = onnx.helper.make_node(
                'Transpose',
                [inputs[1]],
                [transpose_out],
                perm=transpose
            )
            nl.append(n)
            inputs[1] = transpose_out

        # MatMul
        n = onnx.helper.make_node(
            'MatMul',
            inputs,
            func.output
        )
        nl.append(n)

        return nl

    def Softmax(self, opset, func):
        nl = []
        axis = func.softmax_param.axis

        # ReduceMax
        mout = fork_name(func.input[0]) + "_reducemax"
        n = onnx.helper.make_node(
            'ReduceMax',
            [func.input[0]],
            [mout],
            axes=[axis],
            keepdims=True
        )
        nl.append(n)

        # Sub
        sout = fork_name(func.input[0]) + "_sub"
        n = onnx.helper.make_node(
            'Sub',
            [func.input[0], mout],
            [sout],
        )
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Exp
        expout = fork_name(func.input[0]) + "_exp"
        n = onnx.helper.make_node(
            'Exp',
            [sout],
            [expout],
        )
        nl.append(n)

        # ReduceSum
        sumout = fork_name(func.input[0]) + "_reducesum"
        if opset == '13':
            axis_shape = (len([axis]), )
            axis_param_name = fork_name("SoftmaxAxis")
            add_param(self._model_proto.graph, axis_param_name,
                      TensorProto.INT64, axis_shape,
                      np.array([axis]).astype(np.int64).tostring())
            n = onnx.helper.make_node(
                'ReduceSum',
                [expout, axis_param_name],
                [sumout],
                keepdims=1,
                noop_with_empty_axes=0,
            )
        else:
            n = onnx.helper.make_node(
                'ReduceSum',
                [expout],
                [sumout],
                axes=[axis],
                keepdims=True
            )
        nl.append(n)

        # Div
        n = onnx.helper.make_node(
            'Div',
            [expout, sumout],
            [func.output[0]],
        )
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        return nl

    def PReLU(self, func):
        nl = []
        inputs = func.input[:]
        outputs = func.output[:]
        base_axis = func.prelu_param.base_axis
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        slope_shape = list(self._var_dict[func.input[1]].dim[:])
        output_shape = list(self._var_dict[func.output[0]].dim[:])

        if len(slope_shape) == 0:
            slope_shape.append(1)
        if len(slope_shape) != 1:
            raise ValueError("The negative slope must be a 1d")

        if base_axis != 1:
            input0_shape_reshape = np.array(np.concatenate((
                [np.prod(input_shape[:base_axis])], input_shape[base_axis:])))
            rout = fork_name(inputs[0]) + "_reshape"
            n = generate_reshape(self._model_proto.graph, inputs[0],
                                 rout, input0_shape_reshape)
            nl.append(n)
            inputs[0] = rout
            outputs[0] = fork_name(func.output[0]) + "_reshape"
            input_shape = list(input0_shape_reshape)

        s_shape = [1] * len(input_shape)
        s_shape[1] = slope_shape[0]
        proto_s_shape = nnabla_pb2.Shape()
        proto_s_shape.dim.extend(s_shape)
        self._var_dict[inputs[1]] = proto_s_shape

        n = onnx.helper.make_node(
            'PRelu',
            inputs,
            outputs
        )
        nl.append(n)

        if base_axis != 1:
            n = generate_reshape(self._model_proto.graph, outputs[0],
                                 func.output[0], np.array(output_shape))
            nl.append(n)
        return nl

    def Embed(self, func):
        n = onnx.helper.make_node(
            'Gather',
            [func.input[1], func.input[0]],
            func.output,
            axis=0
        )
        self._input_types[func.input[0]] = TensorProto.INT32
        return [n]

    def Swish(self, func):
        # Convert Mul+Sigmoid
        nl = []
        # Sigmoid
        dout = fork_name(func.output[0]) + "_sigmoid"
        n = onnx.helper.make_node("Sigmoid",
                                  func.input,
                                  [dout])
        nl.append(n)

        # Mul
        n = onnx.helper.make_node("Mul",
                                  [func.input[0], dout],
                                  func.output)
        nl.append(n)

        return nl

    def LogSoftmax(self, opset, func):
        # Convert ReduceMax+Sub+Exp+ReduceSum+Log+Sub
        nl = []
        axis = func.log_softmax_param.axis

        # ReduceMax
        mout = fork_name(func.input[0]) + "_reducemax"
        n = onnx.helper.make_node(
            'ReduceMax',
            [func.input[0]],
            [mout],
            axes=[axis],
            keepdims=True
        )
        nl.append(n)

        # Sub
        sout = fork_name(func.input[0]) + "_sub"
        n = onnx.helper.make_node(
            'Sub',
            [func.input[0], mout],
            [sout],
        )
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Exp
        expout = fork_name(func.input[0]) + "_exp"
        n = onnx.helper.make_node(
            'Exp',
            [sout],
            [expout],
        )
        nl.append(n)

        # ReduceSum
        sumout = fork_name(func.input[0]) + "_reducesum"
        if opset == '13':
            axis_shape = (len([axis]), )
            axis_param_name = fork_name("LogSoftmaxAxis")
            add_param(self._model_proto.graph, axis_param_name,
                      TensorProto.INT64, axis_shape,
                      np.array([axis]).astype(np.int64).tostring())
            n = onnx.helper.make_node(
                'ReduceSum',
                [expout, axis_param_name],
                [sumout],
                keepdims=1,
                noop_with_empty_axes=0,
            )
        else:
            n = onnx.helper.make_node(
                'ReduceSum',
                [expout],
                [sumout],
                axes=[axis],
                keepdims=True
            )
        nl.append(n)

        # Log
        logout = fork_name(func.input[0]) + "_log"
        n = onnx.helper.make_node(
            'Log',
            [sumout],
            [logout]
        )
        nl.append(n)

        # Sub
        n = onnx.helper.make_node(
            'Sub',
            [sout, logout],
            func.output
        )
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        return nl

    def CReLU(self, func):
        # Convert Concat+Neg+Relu
        nl = []
        # Neg
        nout = fork_name(func.output[0]) + "_neg"
        n = onnx.helper.make_node("Neg",
                                  [func.input[0]],
                                  [nout])
        nl.append(n)

        # Relu
        rout0 = fork_name(func.output[0]) + "_relu0"
        n = onnx.helper.make_node("Relu",
                                  [func.input[0]],
                                  [rout0])
        nl.append(n)

        # Relu
        rout1 = fork_name(func.output[0]) + "_relu1"
        n = onnx.helper.make_node("Relu",
                                  [nout],
                                  [rout1])
        nl.append(n)

        # Concat
        n = onnx.helper.make_node("Concat",
                                  [rout0, rout1],
                                  func.output,
                                  axis=func.crelu_param.axis)
        nl.append(n)

        return nl

    def ReLU6(self, opset, func):
        # Convert Relu+Constant+Min
        nl = []
        if opset == '6x':
            # Convert Clip
            n = onnx.helper.make_node("Clip",
                                      func.input,
                                      func.output,
                                      min=0.0,
                                      max=6.0)
            nl.append(n)
        else:
            # Convert Relu+Constant+Min
            # Relu
            rout = fork_name(func.output[0]) + "_relu"
            n = onnx.helper.make_node("Relu",
                                      func.input,
                                      [rout])
            nl.append(n)

            # Constant
            constant_six = fork_name("constant")
            input_shape = list(self._var_dict[func.input[0]].dim[:])
            value = [6.0] * np.prod(input_shape)
            c = generate_constant(constant_six, func.name + "_constant_six",
                                  TensorProto.FLOAT, input_shape,
                                  value)
            nl.append(c)

            # Min
            n = onnx.helper.make_node("Min",
                                      [rout, constant_six],
                                      func.output)
            nl.append(n)

        return nl

    def HardTanh(self, func):
        # Convert Constant+Neg+Min+Max
        nl = []
        # Constant
        constant_one = fork_name("constant")
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        value = [1.0] * np.prod(input_shape)
        c = generate_constant(constant_one, func.name + "_constant_one",
                              TensorProto.FLOAT, input_shape,
                              value)
        nl.append(c)

        # Neg
        neg_one = fork_name("constant")
        n = onnx.helper.make_node("Neg",
                                  [constant_one],
                                  [neg_one])
        nl.append(n)

        # Min
        mout = func.output[0] + "_min"
        n = onnx.helper.make_node("Min",
                                  [func.input[0], constant_one],
                                  [mout])
        nl.append(n)

        # Max
        n = onnx.helper.make_node("Max",
                                  [mout, neg_one],
                                  func.output)
        nl.append(n)

        return nl

    def LogSigmoid(self, func):
        # Convert Sigmoid+Log
        nl = []
        # Sigmoid
        sout = fork_name(func.output[0]) + "_sigmoid"
        n = onnx.helper.make_node("Sigmoid",
                                  func.input,
                                  [sout])
        nl.append(n)

        # Log
        n = onnx.helper.make_node("Log",
                                  [sout],
                                  func.output)
        nl.append(n)

        return nl

    def Broadcast(self, func):
        # Convert Constant+Expand
        nl = []
        shape = [self._batch_size if d <
                 0 else d for d in func.broadcast_param.shape.dim]
        # Constant
        constant_newshape = fork_name("constant")
        c = generate_constant(constant_newshape, func.name + "_shape",
                              TensorProto.INT64, [len(shape)],
                              shape)
        nl.append(c)

        # Expand
        n = onnx.helper.make_node("Expand",
                                  [func.input[0], constant_newshape],
                                  func.output)
        nl.append(n)

        return nl

    def CELU(self, func):
        # Convert Neg+Elu+Concat
        nl = []
        alpha = func.celu_param.alpha
        axis = func.celu_param.axis
        # Neg
        neg_out = fork_name(func.input[0]) + "_neg"
        n = onnx.helper.make_node("Neg",
                                  func.input,
                                  [neg_out])
        nl.append(n)

        # Elu
        elu_out0 = fork_name(func.input[0]) + "_elu0"
        n = onnx.helper.make_node("Elu",
                                  func.input,
                                  [elu_out0],
                                  alpha=alpha)
        nl.append(n)

        # Elu
        elu_out1 = fork_name(func.input[0]) + "_elu1"
        n = onnx.helper.make_node("Elu",
                                  [neg_out],
                                  [elu_out1],
                                  alpha=alpha)
        nl.append(n)

        # Concat
        n = onnx.helper.make_node("Concat",
                                  [elu_out0, elu_out1],
                                  func.output,
                                  axis=axis)
        nl.append(n)

        return nl

    def GELU(self, opset, func):
        # Convert Constant+Pow+Mul+Add+Div+Sqrt+Tanh
        nl = []
        # Constant
        constant0 = fork_name("constant")
        c = generate_constant(constant0, func.name + "_constant0",
                              TensorProto.FLOAT, [1],
                              [3])
        nl.append(c)

        # Constant
        constant1 = fork_name("constant")
        c = generate_constant(constant1, func.name + "_constant1",
                              TensorProto.FLOAT, [1],
                              [0.044715])
        nl.append(c)

        # Constant
        constant2 = fork_name("constant")
        c = generate_constant(constant2, func.name + "_constant2",
                              TensorProto.FLOAT, [1],
                              [2])
        nl.append(c)

        # Constant
        constant3 = fork_name("constant")
        c = generate_constant(constant3, func.name + "_constant3",
                              TensorProto.FLOAT, [1],
                              [1])
        nl.append(c)

        # Constant
        constant4 = fork_name("constant")
        c = generate_constant(constant4, func.name + "_constant4",
                              TensorProto.FLOAT, [1],
                              [2/np.pi])
        nl.append(c)

        # Pow
        pow_out = fork_name(func.input[0]) + "_pow"
        n = onnx.helper.make_node("Pow",
                                  [func.input[0], constant0],
                                  [pow_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Mul
        mul_out = fork_name(func.input[0]) + "_mul"
        n = onnx.helper.make_node("Mul",
                                  [pow_out, constant1],
                                  [mul_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Add
        add_out = fork_name(func.input[0]) + "_add"
        n = onnx.helper.make_node("Add",
                                  [func.input[0], mul_out],
                                  [add_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Sqrt
        sqrt_out = fork_name(func.input[0]) + "_sqrt"
        n = onnx.helper.make_node("Sqrt",
                                  [constant4],
                                  [sqrt_out])
        nl.append(n)

        # Mul
        mul_out1 = fork_name(func.input[0]) + "_mul1"
        n = onnx.helper.make_node("Mul",
                                  [add_out, sqrt_out],
                                  [mul_out1])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Tanh
        tanh_out = fork_name(func.input[0]) + "_tanh"
        n = onnx.helper.make_node("Tanh",
                                  [mul_out1],
                                  [tanh_out])
        nl.append(n)

        # Add
        add_out1 = fork_name(func.input[0]) + "_add1"
        n = onnx.helper.make_node("Add",
                                  [tanh_out, constant3],
                                  [add_out1])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Div
        div_out = fork_name(func.input[0]) + "_div"
        n = onnx.helper.make_node("Div",
                                  [func.input[0], constant2],
                                  [div_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Mul
        n = onnx.helper.make_node("Mul",
                                  [div_out, add_out1],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        return nl

    def TanhShrink(self, func):
        # Convert Tanh+Sub
        nl = []
        # Tanh
        tanh_out = fork_name(func.input[0]) + "_tanh"
        n = onnx.helper.make_node("Tanh",
                                  func.input,
                                  [tanh_out])
        nl.append(n)

        # Sub
        n = onnx.helper.make_node("Sub",
                                  [func.input[0], tanh_out],
                                  func.output)
        nl.append(n)

        return nl

    def Arange(self, func):
        # Convert Constant
        start = func.arange_param.start
        stop = func.arange_param.stop
        step = func.arange_param.step
        constant_newshape = fork_name("constant")
        output_shape = list(self._var_dict[func.output[0]].dim[:])
        value = np.arange(start, stop, step).astype(np.float32)
        c = generate_constant(func.output[0], func.name + "_arange",
                              TensorProto.FLOAT, output_shape,
                              value)
        return [c]

    def Tile(self, func):
        # Convert Reshape+Constant+Tile
        nl = []
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        reps = list(func.tile_param.reps)
        input = func.input[0]
        if len(reps) > len(input_shape):
            new_input_shape = np.array(
                [1] * (len(reps) - len(input_shape)) + input_shape)
            input_reshape = fork_name(func.input[0]) + "_reshape"
            n = generate_reshape(self._model_proto.graph, func.input[0], input_reshape,
                                 new_input_shape)
            nl.append(n)
            input = input_reshape
        else:
            reps = (len(input_shape) - len(reps)) * [1] + reps
        # Constant
        constant_reps = fork_name("constant")
        c = generate_constant(constant_reps, func.name + "_reps",
                              TensorProto.INT64, [len(reps)],
                              reps)
        nl.append(c)

        # Tile
        n = onnx.helper.make_node("Tile",
                                  [input, constant_reps],
                                  func.output)
        nl.append(n)

        return nl

    def Sinc(self, func):
        # Convert Constant+Equal+Sin+Div+Where
        nl = []
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        # Constant
        c_zero_out = fork_name("constant")
        c_zero_data = np.zeros(input_shape)
        c = generate_constant(c_zero_out, func.name + "_zero",
                              TensorProto.FLOAT, input_shape,
                              c_zero_data.flatten())
        nl.append(c)

        # Constant
        c_one_out = fork_name("constant")
        c_one_data = np.ones(input_shape)
        c = generate_constant(c_one_out, func.name + "_one",
                              TensorProto.FLOAT, input_shape,
                              c_one_data.flatten())
        nl.append(c)

        # Equal
        equal_out = fork_name(func.output[0]) + "_equal"
        n = onnx.helper.make_node("Equal",
                                  [func.input[0], c_zero_out],
                                  [equal_out])
        nl.append(n)

        # Sin
        sin_out = fork_name(func.output[0]) + "_sin"
        n = onnx.helper.make_node("Sin",
                                  func.input,
                                  [sin_out])
        nl.append(n)

        # Div
        div_out = fork_name(func.output[0]) + "_div"
        n = onnx.helper.make_node("Div",
                                  [sin_out, func.input[0]],
                                  [div_out])
        nl.append(n)

        # Where
        n = onnx.helper.make_node("Where",
                                  [equal_out, c_one_out, div_out],
                                  func.output)
        nl.append(n)

        return nl

    def NotEqual(self, opset, func):
        nl = []
        equal_out = fork_name(func.output[0]) + "_equal"
        n = onnx.helper.make_node("Equal",
                                  func.input,
                                  [equal_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        n = onnx.helper.make_node("Not",
                                  [equal_out],
                                  func.output)
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def GreaterEqual(self, opset, func):
        nl = []
        less_out = fork_name(func.output[0]) + "_less"
        n = onnx.helper.make_node("Less",
                                  func.input,
                                  [less_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        n = onnx.helper.make_node("Not",
                                  [less_out],
                                  func.output)
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def LessEqual(self, opset, func):
        nl = []
        greater_out = fork_name(func.output[0]) + "_greater"
        n = onnx.helper.make_node("Greater",
                                  func.input,
                                  [greater_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        n = onnx.helper.make_node("Not",
                                  [greater_out],
                                  func.output)
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def LogicalAndScalar(self, opset, func):
        nl = []
        scalar = func.logical_and_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.BOOL, [1],
                              [scalar])
        nl.append(c)

        n = onnx.helper.make_node("And",
                                  [func.input[0], constant_scalar],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        self._input_types[func.input[0]] = TensorProto.BOOL
        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def LogicalOrScalar(self, opset, func):
        nl = []
        scalar = func.logical_or_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.BOOL, [1],
                              [scalar])
        nl.append(c)

        n = onnx.helper.make_node("Or",
                                  [func.input[0], constant_scalar],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        self._input_types[func.input[0]] = TensorProto.BOOL
        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def LogicalXorScalar(self, opset, func):
        nl = []
        scalar = func.logical_xor_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.BOOL, [1],
                              [scalar])
        nl.append(c)

        n = onnx.helper.make_node("Xor",
                                  [func.input[0], constant_scalar],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        self._input_types[func.input[0]] = TensorProto.BOOL
        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def EqualScalar(self, opset, func):
        nl = []
        scalar = func.equal_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.FLOAT, [1],
                              [scalar])
        nl.append(c)

        n = onnx.helper.make_node("Equal",
                                  [func.input[0], constant_scalar],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def NotEqualScalar(self, opset, func):
        nl = []
        scalar = func.not_equal_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.FLOAT, [1],
                              [scalar])
        nl.append(c)

        equal_out = fork_name(func.output[0]) + '_equal'
        n = onnx.helper.make_node("Equal",
                                  [func.input[0], constant_scalar],
                                  [equal_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        n = onnx.helper.make_node("Not",
                                  [equal_out],
                                  func.output)
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def GreaterEqualScalar(self, opset, func):
        nl = []
        scalar = func.greater_equal_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.FLOAT, [1],
                              [scalar])
        nl.append(c)

        less_out = fork_name(func.output[0]) + '_less'
        n = onnx.helper.make_node("Less",
                                  [func.input[0], constant_scalar],
                                  [less_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        n = onnx.helper.make_node("Not",
                                  [less_out],
                                  func.output)
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def GreaterScalar(self, opset, func):
        nl = []
        scalar = func.greater_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.FLOAT, [1],
                              [scalar])
        nl.append(c)

        n = onnx.helper.make_node("Greater",
                                  [func.input[0], constant_scalar],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def LessEqualScalar(self, opset, func):
        nl = []
        scalar = func.less_equal_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.FLOAT, [1],
                              [scalar])
        nl.append(c)

        greater_out = fork_name(func.output[0]) + '_greater'
        n = onnx.helper.make_node("Greater",
                                  [func.input[0], constant_scalar],
                                  [greater_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        n = onnx.helper.make_node("Not",
                                  [greater_out],
                                  func.output)
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def LessScalar(self, opset, func):
        nl = []
        scalar = func.less_scalar_param.val
        constant_scalar = fork_name("constant")
        c = generate_constant(constant_scalar, func.name + "_scalar",
                              TensorProto.FLOAT, [1],
                              [scalar])
        nl.append(c)

        n = onnx.helper.make_node("Less",
                                  [func.input[0], constant_scalar],
                                  func.output)
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        self._output_types[func.output[0]] = TensorProto.BOOL

        return nl

    def ReduceSum(self, func):
        nl = []
        axes = func.sum_param.axes
        k = func.sum_param.keep_dims

        axes_shape = (len(axes), )
        axes_param_name = fork_name("SumAxes")
        add_param(self._model_proto.graph, axes_param_name,
                  TensorProto.INT64, axes_shape,
                  np.array(axes).astype(np.int64).tostring())
        n = onnx.helper.make_node(
            'ReduceSum',
            [func.input[0], axes_param_name],
            func.output,
            keepdims=k,
            noop_with_empty_axes=0,
            name=fork_name('ReduceSum'),
        )
        nl.append(n)

        return nl

    def ResetNaN(self, func):
        # Convert Constant+IsNaN+Where
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        val = func.reset_nan_param.val
        nl = []

        # Constant
        val_name = fork_name(func.input[0]) + "_val"
        c = generate_constant(val_name, func.name + "_val",
                              TensorProto.FLOAT, input_shape,
                              [val] * np.prod(input_shape))
        nl.append(c)

        # IsNaN
        isnan_out = fork_name(func.input[0]) + "_isnan"
        n = onnx.helper.make_node(
            "IsNaN",
            func.input,
            [isnan_out]
        )
        nl.append(n)

        # Where
        n = onnx.helper.make_node(
            "Where",
            [isnan_out, val_name, func.input[0]],
            func.output
        )
        nl.append(n)

        return nl

    def ResetInf(self, func):
        # Convert Constant+IsInf+Where
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        val = func.reset_inf_param.val
        nl = []

        # Constant
        val_name = fork_name(func.input[0]) + "_val"
        c = generate_constant(val_name, func.name + "_val",
                              TensorProto.FLOAT, input_shape,
                              [val] * np.prod(input_shape))
        nl.append(c)

        # IsInf
        isinf_out = fork_name(func.input[0]) + "_isinf"
        n = onnx.helper.make_node(
            "IsInf",
            func.input,
            [isinf_out]
        )
        nl.append(n)

        # Where
        n = onnx.helper.make_node(
            "Where",
            [isinf_out, val_name, func.input[0]],
            func.output
        )
        nl.append(n)

        return nl

    def Interpolate_10(self, func):
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        output_shape = list(self._var_dict[func.output[0]].dim[:])
        input = func.input[0]
        output = func.output[0]
        output_size = func.interpolate_param.output_size
        mode = func.interpolate_param.mode
        align_corners = func.interpolate_param.align_corners
        half_pixel = func.interpolate_param.half_pixel
        half_pixel_for_nn = func.interpolate_param.half_pixel_for_nn
        scale = []
        nl = []

        if (mode == "linear" and (align_corners or half_pixel)) or \
                (mode == "nearest" and (align_corners or half_pixel or half_pixel_for_nn)):
            raise ValueError(
                "align_corners, half_pixel, and half_pixel_for_nn must be all false.")

        if len(input_shape) == 4:
            scale = [float(output_shape[i]) / float(input_shape[i])
                     for i in range(4)]
        else:
            diff = len(input_shape) - 4
            if diff > 0:
                diff += 1
                new_input_shape = [
                    np.prod(input_shape[:diff])] + input_shape[diff:]
                new_output_shape = [new_input_shape[0]] + output_shape[diff:]
            else:
                new_input_shape = [1] * -diff + input_shape
                new_output_shape = [1] * -diff + output_shape
            scale = [float(new_output_shape[i]) /
                     float(new_input_shape[i]) for i in range(4)]

            reshape_out = fork_name(func.input[0]) + '_reshape'
            n = generate_reshape(self._model_proto.graph, func.input[0], reshape_out,
                                 np.array(new_input_shape))
            nl.append(n)
            input = reshape_out
            output = fork_name(func.input[0]) + '_resize'

        scale_out = fork_name(func.input[0]) + '_scale'
        add_param(self._model_proto.graph, scale_out,
                  TensorProto.FLOAT, [4], np.array(scale, dtype=np.float32).tostring())

        n = onnx.helper.make_node(
            "Resize",
            [input, scale_out],
            [output],
            mode=mode,
        )
        nl.append(n)

        if len(input_shape) != 4:
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 np.array(output_shape))
            nl.append(n)
        return nl

    def Interpolate_11(self, func):
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        output_shape = list(self._var_dict[func.output[0]].dim[:])
        input = func.input[0]
        output = func.output[0]
        output_size = func.interpolate_param.output_size
        mode = func.interpolate_param.mode
        align_corners = func.interpolate_param.align_corners
        half_pixel = func.interpolate_param.half_pixel
        half_pixel_for_nn = func.interpolate_param.half_pixel_for_nn
        scale = []
        nl = []

        if align_corners and half_pixel:
            raise ValueError(
                "Currently (align_corners == true) and (half_pixel == true) is not supported.")

        roi_out = fork_name(func.input[0]) + '_roi'
        add_param(self._model_proto.graph, roi_out,
                  TensorProto.FLOAT, [8], np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32).tostring())

        if len(input_shape) == 4:
            scale = [float(output_shape[i]) / float(input_shape[i])
                     for i in range(4)]
        else:
            diff = len(input_shape) - 4
            if diff > 0:
                diff += 1
                new_input_shape = [
                    np.prod(input_shape[:diff])] + input_shape[diff:]
                new_output_shape = [new_input_shape[0]] + output_shape[diff:]
            else:
                new_input_shape = [1] * -diff + input_shape
                new_output_shape = [1] * -diff + output_shape
            scale = [float(new_output_shape[i]) /
                     float(new_input_shape[i]) for i in range(4)]

            reshape_out = fork_name(func.input[0]) + '_reshape'
            n = generate_reshape(self._model_proto.graph, func.input[0], reshape_out,
                                 np.array(new_input_shape))
            nl.append(n)
            input = reshape_out
            output = fork_name(func.input[0]) + '_resize'

        scale_out = fork_name(func.input[0]) + '_scale'
        add_param(self._model_proto.graph, scale_out,
                  TensorProto.FLOAT, [4], np.array(scale, dtype=np.float32).tostring())

        n = onnx.helper.make_node(
            "Resize",
            [input, roi_out, scale_out],
            [output],
            mode=mode,
        )
        if mode == "nearest":
            if half_pixel_for_nn:
                c = onnx.helper.make_attribute(
                    "coordinate_transformation_mode", 'tf_half_pixel_for_nn')
                n.attribute.extend([c])
            else:
                if align_corners and not half_pixel:
                    c = onnx.helper.make_attribute(
                        "coordinate_transformation_mode", 'align_corners')
                    n.attribute.extend([c])
                elif half_pixel and not align_corners:
                    c = onnx.helper.make_attribute(
                        "coordinate_transformation_mode", 'pytorch_half_pixel')
                    n.attribute.extend([c])
                elif not half_pixel and not align_corners:
                    c = onnx.helper.make_attribute(
                        "coordinate_transformation_mode", 'asymmetric')
                    n.attribute.extend([c])
            c = onnx.helper.make_attribute(
                "nearest_mode", 'floor')
            n.attribute.extend([c])
        elif mode == "linear":
            if align_corners and not half_pixel:
                c = onnx.helper.make_attribute(
                    "coordinate_transformation_mode", 'align_corners')
                n.attribute.extend([c])
            elif half_pixel and not align_corners:
                c = onnx.helper.make_attribute(
                    "coordinate_transformation_mode", 'pytorch_half_pixel')
                n.attribute.extend([c])
            elif not half_pixel and not align_corners:
                c = onnx.helper.make_attribute(
                    "coordinate_transformation_mode", 'asymmetric')
                n.attribute.extend([c])
        nl.append(n)

        if len(input_shape) != 4:
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 np.array(output_shape))
            nl.append(n)
        return nl

    def Pad(self, opset, func):
        input_shape = list(self._var_dict[func.input[0]].dim[:])
        inputs = func.input[:]
        pp = func.pad_param
        mode_conv = {
            "constant": "constant",
            "replicate": "edge",
            "reflect": "reflect"
        }
        # separate pad values to match ONNX format
        # (S0,E0,S1,E1) => (S0,S1,E0,E1)
        dim = len(pp.pad_width) // 2
        zero_dim_num = len(input_shape) - dim
        it = iter(pp.pad_width)
        starts = [0] * zero_dim_num
        ends = [0] * zero_dim_num
        for x in it:
            starts.append(x)
            ends.append(next(it))
        starts.extend(ends)
        pad_nodes = generate_pad(
            func.input[0], func.output[0], pp.mode, starts, pp.constant_value, opset)
        return pad_nodes

    def Tanh(self, func):
        # add attr alpha&beta for Tanh, This is the attr necessary for SNPE to handle Tanh.
        n = onnx.helper.make_node(
            "Tanh",
            func.input,
            func.output,
            alpha=1.0,
            beta=1.0)
        return [n]

    def DequantizeLinear(self, func):
        nl = []
        self._input_types[func.input[0]] = self._quantize_dtype
        self._input_types[func.input[2]] = self._quantize_dtype
        for p in func.input[1:]:
            shape = np.prod(list(self._var_dict[p].dim[:]))
            new_shape = nnabla_pb2.Shape()
            new_shape.dim.extend([shape])
            self._var_dict[p] = new_shape

        n = onnx.helper.make_node(
            "DequantizeLinear",
            func.input,
            func.output
        )
        nl.append(n)
        return nl

    def DequantizeLinear_trt(self, func):
        nl = []
        self._input_types[func.input[0]] = self._quantize_dtype
        self._input_types[func.input[2]] = self._quantize_dtype
        self._scale_cnt[func.input[1]] += 1

        if func.input[1] not in self._parameters or func.input[2] not in self._parameters:
            raise ValueError(
                "The type of scale and zero_point must be parameter")

        if self._scale_cnt[func.input[1]] > 1:
            scale_name = fork_name(func.input[1])
            scale = [1/d for d in self._parameters[func.input[1]].data]
            p = self._nnp.parameter.add()
            p.variable_name = scale_name
            p.shape.dim.extend([len(scale)])
            p.data.extend(scale)
            self._var_dict[scale_name] = self._var_dict[func.input[1]]
            func.input[1] = scale_name
        else:
            scale = [1/d for d in self._parameters[func.input[1]].data]
            for p in self._nnp.parameter[:]:
                if p.variable_name == func.input[1]:
                    self._nnp.parameter.remove(p)
            p = self._nnp.parameter.add()
            p.variable_name = func.input[1]
            p.shape.dim.extend([len(scale)])
            p.data.extend(scale)

        for p in func.input[1:]:
            shape = np.prod(list(self._var_dict[p].dim[:]))
            new_shape = nnabla_pb2.Shape()
            new_shape.dim.extend([shape])
            self._var_dict[p] = new_shape

        n = onnx.helper.make_node(
            "DequantizeLinear",
            func.input,
            func.output
        )
        nl.append(n)
        return nl

    def QuantizeLinear(self, func):
        INT_TO_TENSOR_TYPE = {
            # Currently, ONNX's QuantizeLinear only supports int8 and uint8
            1: TensorProto.INT8,
            2: TensorProto.UINT8,
        }
        nl = []
        inputs = func.input[:]
        round_mode = func.quantize_linear_param.round_mode
        narrow_range = func.quantize_linear_param.narrow_range
        dtype = func.quantize_linear_param.dtype

        if round_mode == "HALF_AWAY_FROM_ZERO":
            raise ValueError(
                "Currently, round_mode is {} not supported.".format(round_mode))

        if narrow_range:
            raise ValueError("Currently, narrow_range is True not supported.")

        if dtype not in INT_TO_TENSOR_TYPE:
            raise ValueError(
                "Currently, dtype is {} not supported.".format(dtype))
        else:
            self._input_types[func.input[2]] = INT_TO_TENSOR_TYPE[dtype]
            self._output_types[func.output[0]] = INT_TO_TENSOR_TYPE[dtype]
            self._quantize_dtype = INT_TO_TENSOR_TYPE[dtype]

        for p in func.input[1:]:
            shape = np.prod(list(self._var_dict[p].dim[:]))
            new_shape = nnabla_pb2.Shape()
            new_shape.dim.extend([shape])
            self._var_dict[p] = new_shape

        self._scale_cnt[func.input[1]] += 1

        n = onnx.helper.make_node(
            "QuantizeLinear",
            inputs,
            func.output
        )
        nl.append(n)
        return nl

    def WeightStandardization(self, opset, func):
        nl = []
        pp = func.weight_standardization_param
        eps = pp.eps
        channel_axis = pp.channel_axis
        w_shape = list(self._var_dict[func.input[0]].dim[:])
        ndim = len(w_shape)

        if not hasattr(channel_axis, "__iter__"):
            channel_axis = [channel_axis]
        axes = tuple([i for i in range(ndim) if i not in channel_axis])

        # ReduceMean
        mean_out = fork_name(func.input[0]) + "_reducemean"
        n = onnx.helper.make_node(
            'ReduceMean',
            [func.input[0]],
            [mean_out],
            axes=axes,
            keepdims=True
        )
        nl.append(n)

        # w_var = w.var(axis=axes, keepdims=True)
        var_nl, var_out = get_matrix_variance(opset, self._model_proto.graph,
                                              func.input[0], func.name, mean_out, axes, w_shape)
        nl.extend(var_nl)

        constant0 = fork_name("constant")
        c = generate_constant(constant0, func.name + "_constant0",
                              TensorProto.FLOAT, [1],
                              [eps])
        nl.append(c)

        constant1 = fork_name("constant")
        c = generate_constant(constant1, func.name + "_constant1",
                              TensorProto.FLOAT, [1],
                              [0.5])
        nl.append(c)

        # Sub, (w - w_mean)
        sub_out = fork_name(func.input[0]) + "_sub"
        n = onnx.helper.make_node("Sub",
                                  [func.input[0], mean_out],
                                  [sub_out]
                                  )
        nl.append(n)

        # Add, (w_var + eps)
        add_out = fork_name(func.output[0]) + "_add"
        n = onnx.helper.make_node(
            "Add",
            [var_out, constant0],
            [add_out]
        )
        nl.append(n)

        # Pow, (w_var + eps) ** 0.5
        pow_out = fork_name(func.input[0]) + "_pow"
        n = onnx.helper.make_node("Pow",
                                  [add_out, constant1],
                                  [pow_out])
        nl.append(n)

        # Div, (w - w_mean) / (w_var + eps) ** 0.5
        dout = fork_name(func.output[0]) + "_div"
        n = onnx.helper.make_node("Div",
                                  [sub_out, pow_out],
                                  func.output)
        nl.append(n)

        return nl

    def LayerNormalization(self, opset, func):
        nl = []
        pp = func.layer_normalization_param
        batch_axis = list(pp.batch_axis)
        eps = pp.eps
        no_scale = pp.no_scale
        no_bias = pp.no_bias

        if not hasattr(batch_axis, "__iter__"):
            batch_axis = [batch_axis]

        x_shape = list(self._var_dict[func.input[0]].dim[:])
        ndim = len(x_shape)
        axes = tuple([i for i in range(ndim) if i not in batch_axis])

        beta = None
        gamma = None

        if len(func.input) == 2:
            if func.input[1] == 'gamma':
                gamma = func.input[1]
            elif func.input[1] == 'beta':
                beta = func.input[1]
        elif len(func.input) == 3:
            beta = func.input[1]
            gamma = func.input[2]

        constant0 = fork_name("constant")
        c = generate_constant(constant0, func.name + "_constant0",
                              TensorProto.FLOAT, [1],
                              [eps])
        nl.append(c)

        constant1 = fork_name("constant")
        c = generate_constant(constant1, func.name + "_constant1",
                              TensorProto.FLOAT, [1],
                              [0.5])
        nl.append(c)

        # ReduceMean
        mean_out = fork_name(func.input[0]) + "_reducemean"
        n = onnx.helper.make_node(
            'ReduceMean',
            [func.input[0]],
            [mean_out],
            axes=axes,
            keepdims=True
        )
        nl.append(n)

        # x_var = x.var(axis=axes, keepdims=True)
        var_nl, var_out = get_matrix_variance(opset, self._model_proto.graph,
                                              func.input[0], func.name, mean_out, axes, x_shape)
        nl.extend(var_nl)

        norm_nl = get_normalization_norm(
            func, mean_out, var_out, beta, gamma, constant0, constant1)
        nl.extend(norm_nl)

        return nl

    def InstanceNormalization(self, opset, func):
        nl = []
        pp = func.instance_normalization_param
        channel_axis = pp.channel_axis
        batch_axis = list(pp.batch_axis)
        eps = pp.eps
        no_scale = pp.no_scale
        no_bias = pp.no_bias
        beta = None
        gamma = None

        if len(func.input) == 2:
            if func.input[1] == 'gamma':
                gamma = func.input[1]
            elif func.input[1] == 'beta':
                beta = func.input[1]
        elif len(func.input) == 3:
            beta = func.input[1]
            gamma = func.input[2]

        # ignore_axes = _force_list(batch_axis) + [channel_axis, ]
        if not hasattr(batch_axis, "__iter__"):
            batch_axis = [batch_axis]
        ignore_axes = batch_axis + [channel_axis, ]

        # axes = tuple(_get_axes_excluding(len(x.shape), ignore_axes))
        x_shape = list(self._var_dict[func.input[0]].dim[:])
        ndim = len(x_shape)
        axes = tuple([i for i in range(ndim) if i not in ignore_axes])

        constant0 = fork_name("constant")
        c = generate_constant(constant0, func.name + "_constant0",
                              TensorProto.FLOAT, [1],
                              [eps])
        nl.append(c)

        constant1 = fork_name("constant")
        c = generate_constant(constant1, func.name + "_constant1",
                              TensorProto.FLOAT, [1],
                              [0.5])
        nl.append(c)

        if not axes:
            # Reshape
            rout = fork_name(func.input[0]) + "_reshape"
            n = generate_reshape(self._model_proto.graph, func.input[0], rout,
                                 np.array(x_shape + [1]))
            nl.append(n)
            axes = (ndim,)

            # ReduceMean
            mean_out_ = fork_name(func.input[0]) + "_reducemean"
            n = onnx.helper.make_node(
                'ReduceMean',
                [rout],
                [mean_out_],
                axes=axes,
                keepdims=True
            )
            nl.append(n)

            mean_out = fork_name(func.input[0]) + "_reshape"
            n = generate_reshape(self._model_proto.graph, mean_out_, mean_out,
                                 np.array(x_shape))
            nl.append(n)

            # x_var = x.var(axis=axes, keepdims=True)
            var_nl, var_out_ = get_matrix_variance(opset, self._model_proto.graph,
                                                   rout, func.name, mean_out_, axes, (x_shape + [1]))
            nl.extend(var_nl)

            var_out = fork_name(func.input[0]) + "_reshape"
            n = generate_reshape(self._model_proto.graph, var_out_, var_out,
                                 np.array(x_shape))
            nl.append(n)

        else:
            # ReduceMean
            mean_out = fork_name(func.input[0]) + "_reducemean"
            n = onnx.helper.make_node(
                'ReduceMean',
                [func.input[0]],
                [mean_out],
                axes=axes,
                keepdims=True
            )
            nl.append(n)

            # x_var = x.var(axis=axes, keepdims=True)
            var_nl, var_out = get_matrix_variance(opset, self._model_proto.graph,
                                                  func.input[0], func.name, mean_out, axes, x_shape)
            nl.extend(var_nl)

        norm_nl = get_normalization_norm(
            func, mean_out, var_out, beta, gamma, constant0, constant1)
        nl.extend(norm_nl)

        return nl

    def WeightNormalization(self, opset, func):
        nl = []
        w_shape = list(self._var_dict[func.input[0]].dim[:])
        pp = func.weight_normalization_param
        dim = pp.dim
        eps = pp.eps
        axes = tuple([a for a in range(len(w_shape)) if a != dim])

        constant0 = fork_name("constant")
        c = generate_constant(constant0, func.name + "_constant0",
                              TensorProto.FLOAT, [1],
                              [2])
        nl.append(c)

        constant1 = fork_name("constant")
        c = generate_constant(constant1, func.name + "_constant1",
                              TensorProto.FLOAT, [1],
                              [eps])
        nl.append(c)

        constant2 = fork_name("constant")
        c = generate_constant(constant2, func.name + "_constant2",
                              TensorProto.FLOAT, [1],
                              [-0.5])
        nl.append(c)

        # Pow, w ** 2
        pow_out1 = fork_name(func.input[0]) + "_pow"
        n = onnx.helper.make_node("Pow",
                                  [func.input[0], constant0],
                                  [pow_out1])
        nl.append(n)

        # ReduceSum, np.sum(w ** 2, axes, keepdims=True)
        sum_out = fork_name(func.input[0]) + "_sum"
        if opset == '13':
            axes_shape = (len(axes), )
            axes_param_name = fork_name("WnReduceSumAxes")
            add_param(self._model_proto.graph, axes_param_name,
                      TensorProto.INT64, axes_shape,
                      np.array(axes).astype(np.int64).tostring())
            n = onnx.helper.make_node(
                'ReduceSum',
                [pow_out1, axes_param_name],
                [sum_out],
                keepdims=1,
                noop_with_empty_axes=0,
            )
        else:
            n = onnx.helper.make_node(
                'ReduceSum',
                [pow_out1],
                [sum_out],
                axes=axes,
                keepdims=True
            )
        nl.append(n)

        # Add, (np.sum(w ** 2, axes, keepdims=True) + eps)
        add_out = fork_name(func.input[0]) + "_add"
        n = onnx.helper.make_node(
            "Add",
            [sum_out, constant1],
            [add_out]
        )
        nl.append(n)

        # Pow, (np.sum(w ** 2, axes, keepdims=True) + eps) ** (-0.5)
        pow_out2 = fork_name(func.input[0]) + "_pow"
        n = onnx.helper.make_node("Pow",
                                  [add_out, constant2],
                                  [pow_out2])
        nl.append(n)

        # Reshape, g.reshape(rshape)
        input_shape = [1 if i != dim else s for i, s in enumerate(w_shape)]
        input_shape_reshape = np.array(input_shape)
        rout = fork_name(func.input[1]) + "_reshape"
        n = generate_reshape(self._model_proto.graph, func.input[1], rout,
                             input_shape_reshape)
        nl.append(n)

        # Mul, g * w * n
        mul_out1 = fork_name(func.input[0]) + "_mul"
        n = onnx.helper.make_node(
            "Mul",
            [rout, func.input[0]],
            [mul_out1]
        )
        nl.append(n)

        n = onnx.helper.make_node(
            "Mul",
            [mul_out1, pow_out2],
            func.output
        )
        nl.append(n)

        return nl

    def GroupNormalization(self, opset, func):
        def cycle_index(index, list_len):
            mod = index % list_len
            return mod if mod >= 0 else mod + list_len

        nl = []

        # set input
        beta = None
        gamma = None
        if len(func.input) == 2:
            if func.input[1] == 'gamma':
                gamma = func.input[1]
            elif func.input[1] == 'beta':
                beta = func.input[1]
        elif len(func.input) == 3:
            beta = func.input[1]
            gamma = func.input[2]

        # set pp
        pp = func.group_normalization_param
        gn = pp.num_groups if hasattr(pp, 'num_groups') else 1
        channel_axis = [pp.channel_axis] if hasattr(
            pp, 'channel_axis') else [1]
        batch_axis = [pp.batch_axis] if hasattr(pp, 'batch_axis') else [0]
        eps = pp.eps if hasattr(pp, 'eps') else 1e-05
        no_scale = pp.no_scale if hasattr(pp, 'no_scale') else False
        no_bias = pp.no_bias if hasattr(pp, 'no_bias') else False

        channel_axis = np.array(channel_axis).flatten()
        batch_axis = np.array(batch_axis).flatten()

        x_shape = list(self._var_dict[func.input[0]].dim[:])
        len_x_shape = len(x_shape)

        for i in range(len(channel_axis)):
            channel_axis[i] = cycle_index(channel_axis[i], len_x_shape)
        for i in range(len(batch_axis)):
            batch_axis[i] = cycle_index(batch_axis[i], len_x_shape)

        channel_num = np.prod([x_shape[i] for i in channel_axis])
        if channel_num != channel_num // gn * gn:
            raise ValueError(
                "The channel dim of 'x' must be integer multiple of `num_groups`.")

        # set x_reshape set axes
        x_shape_reshape = []
        axes = []
        i_num = 0
        for i in range(len(x_shape)):
            if i == channel_axis[0]:
                x_shape_reshape.extend([gn, x_shape[i] // gn])
                i_num += 1
                axes.append(i + i_num)
            elif i in batch_axis:
                x_shape_reshape.extend([x_shape[i]])
            else:
                x_shape_reshape.extend([x_shape[i]])
                axes.append(i + i_num)

        reshape_out = fork_name(func.input[0]) + "_reshape"
        n = generate_reshape(self._model_proto.graph, func.input[0], reshape_out,
                             np.array(x_shape_reshape))
        nl.append(n)

        # ReduceMean
        mean_out = fork_name(func.input[0]) + "_reducemean"
        n = onnx.helper.make_node(
            'ReduceMean',
            [reshape_out],
            [mean_out],
            axes=axes,
            keepdims=True
        )
        nl.append(n)

        # Sub, (x - m)
        sub_out = fork_name(func.input[0]) + "_sub"
        n = onnx.helper.make_node("Sub",
                                  [reshape_out, mean_out],
                                  [sub_out]
                                  )
        nl.append(n)

        # (x-m)(x-m)
        mul_out_m = fork_name(func.input[0]) + "_mul"
        n = onnx.helper.make_node(
            'Mul',
            [sub_out, sub_out],
            [mul_out_m],
        )
        nl.append(n)

        # x_var
        var_out_ = fork_name(func.input[0]) + "_reducemean"
        n = onnx.helper.make_node(
            'ReduceMean',
            [mul_out_m],
            [var_out_],
            axes=axes,
            keepdims=True
        )
        nl.append(n)

        constant0 = fork_name("constant")
        c = generate_constant(constant0, func.name + "_constant0",
                              TensorProto.FLOAT, [1],
                              [eps])
        nl.append(c)

        # Add, (x_var + eps)
        add_out = fork_name(func.input[0]) + "_add"
        n = onnx.helper.make_node(
            "Add",
            [var_out_, constant0],
            [add_out]
        )
        nl.append(n)

        # sqrt, (x_var + eps) ** 0.5
        sqrt_out = fork_name(func.input[0]) + "_sqrt"
        n = onnx.helper.make_node("Sqrt",
                                  [add_out],
                                  [sqrt_out])
        nl.append(n)

        # Div
        dout = fork_name(func.input[0]) + "_div"
        n = onnx.helper.make_node("Div",
                                  [sub_out, sqrt_out],
                                  [dout])
        nl.append(n)

        # Reshape
        reshape1_out = fork_name(func.input[0]) + "_reshape"
        n = generate_reshape(self._model_proto.graph, dout, reshape1_out,
                             np.array(x_shape))
        nl.append(n)

        beta_gamma_dims = np.prod([x_shape[i] for i in channel_axis])
        beta_gamma_shape = [x_shape[i] if i in channel_axis else 1
                            for i in range(len(x_shape))]

        if None == gamma:
            gamma = 'gamma'
            c = generate_constant(gamma, func.name + "_constant_g",
                                  TensorProto.FLOAT, beta_gamma_shape,
                                  [1.0] * beta_gamma_dims)
            nl.append(c)
        if None == beta:
            beta = 'beta'
            c = generate_constant(beta, func.name + "_constant_b",
                                  TensorProto.FLOAT, beta_gamma_shape,
                                  [0.0] * beta_gamma_dims)
            nl.append(c)

        # Mul gamma
        mul_out = fork_name(func.input[0]) + "_mul"
        n = onnx.helper.make_node(
            'Mul',
            [reshape1_out, gamma],
            [mul_out],
        )
        nl.append(n)

        # Add, beta
        add_beta_out = fork_name(func.input[0]) + "_add"
        n = onnx.helper.make_node(
            "Add",
            [mul_out, beta],
            [func.output[0]]
        )
        nl.append(n)

        return nl

    def SpectralNorm(self, func):
        nl = []
        pp = func.spectral_norm_param
        dim = pp.dim
        itr = pp.itr
        eps = pp.eps
        test = pp.test
        w_shape = list(self._var_dict[func.input[0]].dim[:])
        u_shape = list(self._var_dict[func.input[1]].dim[:])
        constant0 = fork_name("constant")
        c = generate_constant(constant0, func.name + "_constant0",
                              TensorProto.FLOAT, [1],
                              [2])
        nl.append(c)

        constant1 = fork_name("constant")
        c = generate_constant(constant1, func.name + "_constant1",
                              TensorProto.FLOAT, [1],
                              [eps])
        nl.append(c)

        constant2 = fork_name("constant")
        c = generate_constant(constant2, func.name + "_constant2",
                              TensorProto.FLOAT, [1], [0])
        nl.append(c)

        if dim != 0:
            dims_transpose = [dim] + \
                [i for i in range(len(w_shape)) if i != dim]
            transpose_out = fork_name(func.input[0]) + "_transpose"
            n = onnx.helper.make_node(
                'Transpose',
                [func.input[0]],
                [transpose_out],
                perm=dims_transpose
            )
            nl.append(n)
            func.input[0] = transpose_out
            w_data = np.random.randn(*w_shape)
            w_data = w_data.transpose(*dims_transpose)
            w_shape = w_data.shape

        d0, d1 = w_shape[0], np.prod(w_shape[1:])
        reshape_w_out = fork_name(func.input[0]) + "_reshape"
        w_shape_ = [d0, d1]
        n = generate_reshape(self._model_proto.graph,
                             func.input[0], reshape_w_out, np.array(w_shape_))
        nl.append(n)
        func.input[0] = reshape_w_out

        for i in range(itr):
            # v = np.dot(w.T, u)
            bias3_shape = [int(np.prod(w_shape_) / w_shape_[0]), 1]
            bias3 = np.zeros(bias3_shape)
            constant3 = fork_name("constant")
            c = generate_constant(constant3, func.name + "_constant3",
                                  TensorProto.FLOAT, bias3_shape,
                                  bias3)
            nl.append(c)

            reshape_u_out = fork_name(func.input[1]) + "_reshape"
            reshape_u_shape = [u_shape[0], 1]
            n = generate_reshape(
                self._model_proto.graph, func.input[1], reshape_u_out, np.array(reshape_u_shape))
            nl.append(n)

            gemm_out1 = fork_name(func.output[0]) + "_gemm"
            n = onnx.helper.make_node(
                "Gemm",
                [func.input[0], reshape_u_out, constant3],
                [gemm_out1],
                transA=1,
                transB=0,
            )
            nl.append(n)

            # v ** 2
            pow_out1 = fork_name(func.input[0]) + "_pow"
            n = onnx.helper.make_node(
                'Pow',
                [gemm_out1, constant0],
                [pow_out1]
            )
            nl.append(n)

            # np.sum(v ** 2)
            sum_out1 = fork_name(func.input[0]) + "_sum"
            n = onnx.helper.make_node(
                'ReduceSum',
                [pow_out1],
                [sum_out1]
            )
            nl.append(n)

            # Add, np.sum(v ** 2) + eps
            add_out1 = fork_name(func.output[0]) + "_add"
            n = onnx.helper.make_node(
                "Add",
                [sum_out1, constant1],
                [add_out1]
            )
            nl.append(n)

            # np.sqrt(np.sum(v ** 2) + eps)
            sqrt_out1 = fork_name(func.input[0]) + "_sqrt"
            n = onnx.helper.make_node(
                'Sqrt',
                [add_out1],
                [sqrt_out1]
            )
            nl.append(n)

            # v = v / np.sqrt(np.sum(v ** 2) + eps)
            div_out1 = fork_name(func.input[0]) + "_div"
            n = onnx.helper.make_node(
                'Div',
                [gemm_out1, sqrt_out1],
                [div_out1]
            )
            nl.append(n)

            # u = np.dot(w, v)
            constant4 = fork_name("constant")
            bias4_shape = [int(np.prod(np.prod(w_shape_)) / w_shape_[-1]), 1]
            bias4 = np.zeros(bias4_shape)
            c = generate_constant(constant4, func.name + "_constant4",
                                  TensorProto.FLOAT, bias4_shape,
                                  bias4)
            nl.append(c)

            reshape_v_out = fork_name(func.input[1]) + "_reshape"
            reshape_v_shape = bias3_shape
            n = generate_reshape(
                self._model_proto.graph, div_out1, reshape_v_out, np.array(reshape_v_shape))
            nl.append(n)

            gemm_out2 = fork_name(func.input[0]) + "_gemm"
            n = onnx.helper.make_node(
                'Gemm',
                [func.input[0], reshape_v_out, constant4],
                [gemm_out2])
            nl.append(n)

            # u ** 2
            pow_out2 = fork_name(func.input[0]) + "_pow"
            n = onnx.helper.make_node(
                'Pow',
                [gemm_out2, constant0],
                [pow_out2]
            )
            nl.append(n)

            # np.sum(u ** 2)
            sum_out2 = fork_name(func.input[0]) + "_sum"
            n = onnx.helper.make_node(
                'ReduceSum',
                [pow_out2],
                [sum_out2]
            )
            nl.append(n)

            # Add, (w_var + eps)
            add_out2 = fork_name(func.output[0]) + "_add"
            n = onnx.helper.make_node(
                "Add",
                [sum_out2, constant1],
                [add_out2]
            )
            nl.append(n)

            # np.sqrt(np.sum(u ** 2) + eps)
            sqrt_out2 = fork_name(func.input[0]) + "_sqrt"
            n = onnx.helper.make_node(
                'Sqrt',
                [add_out2],
                [sqrt_out2]
            )
            nl.append(n)

            # u = u / np.sqrt(np.sum(u ** 2) + eps)
            div_out2 = fork_name(func.input[0]) + "_div"
            n = onnx.helper.make_node(
                'Div',
                [gemm_out2, sqrt_out2],
                [div_out2]
            )
            nl.append(n)

            func.input[1] = div_out2

        # np.dot(w, v)
        gemm_out3 = fork_name(func.input[0]) + "_gemm"
        n = onnx.helper.make_node(
            'Gemm',
            [func.input[0], reshape_v_out, constant4],
            [gemm_out3]
        )
        nl.append(n)

        gemm_out4 = fork_name(func.input[0]) + "_gemm"
        n = onnx.helper.make_node(
            'Gemm',
            [div_out2, gemm_out3, constant2],
            [gemm_out4],
            transA=1,
            transB=0
        )
        nl.append(n)

        # w_sn = w / sigma
        div_out3 = fork_name(func.input[0]) + "_div"
        n = onnx.helper.make_node(
            'Div',
            [func.input[0], gemm_out4],
            [div_out3]
        )
        nl.append(n)

        reshape_out = fork_name(func.input[0]) + "_reshape"
        reshape = np.array(w_shape)
        n = generate_reshape(self._model_proto.graph,
                             div_out3, reshape_out, reshape)
        nl.append(n)

        if dim != 0:
            dims_transpose = [i for i in range(1, dim + 1)] \
                             + [0] + [i for i in range(dim + 1, len(w_shape))]
            n = onnx.helper.make_node(
                'Transpose',
                [reshape_out],
                [func.output[0]],
                perm=dims_transpose
            )
            nl.append(n)
        else:
            nl[-1].output[0] = func.output[0]
        return nl

    def set_network(self):
        if len(self._nnp.executor) != 1:
            raise ValueError(
                "NNP with only a single executor is currently supported")
        exe = self._nnp.executor[0]

        net = None
        for n in self._nnp.network:
            if n.name == exe.network_name:
                net = n
        if net is None:
            raise ValueError(
                "Executor network [{}] is not found in this NNP.".format(exe.network_name))
        self._net = net
        self._executor = exe
        return net

    def set_shape_all(self):
        bs = self._batch_size
        if bs < 0:
            bs = self._net.batch_size
        self._batch_size = bs
        # store all variable shape info to use later
        for v in self._net.variable:
            self._var_dict[v.name] = replace_negative_size_with_batch_size(
                v.shape, bs)

        for p in self._nnp.parameter:
            self._parameters[p.variable_name] = p

    def set_variables(self):
        exe = self._executor
        graph = self._model_proto.graph
        for param in self._nnp.parameter:
            if param.variable_name in self._var_dict:
                init = graph.initializer.add()
                init.name = param.variable_name
                dims = [d for d in self._var_dict[init.name].dim]
                init.dims.extend(dims)
                t = get_tensor_type(param.variable_name, self._input_types)
                init.data_type = t
                init.raw_data = np.array(
                    param.data, dtype=TENSOR_TYPE_TO_DTYPE[t]).tostring()
            else:
                print("Not in: {}".format(param.variable_name))

        for iv in exe.data_variable:
            i = graph.input.add()
            i.name = iv.variable_name
            i.type.tensor_type.elem_type = get_tensor_type(
                iv.variable_name, self._input_types)
            dims = [create_dim(d)
                    for d in self._var_dict[iv.variable_name].dim]
            i.type.tensor_type.shape.dim.extend(dims)

        # Add only the final output of the graph as output
        for ov in exe.output_variable:
            o = graph.output.add()
            o.name = ov.variable_name
            o.type.tensor_type.elem_type = get_tensor_type(
                ov.variable_name, self._output_types)
            dims = [create_dim(d)
                    for d in self._var_dict[ov.variable_name].dim]
            o.type.tensor_type.shape.dim.extend(dims)

        for gv in exe.generator_variable:
            init = graph.initializer.add()
            init.name = gv.variable_name
            init.data_type = get_tensor_type(
                gv.variable_name, self._input_types)
            dims = self._var_dict[gv.variable_name].dim
            init.dims.extend(dims)
            init.raw_data = generate_value(
                gv.type, dims, init.data_type, gv.multiplier)

    def set_nodes(self, func):
        """Convert a function to a node or a group of nodes"""
        op_type = self.nnabla_function_type_to_onnx_optype.get(func.type)
        if op_type is None:
            raise ValueError(
                "function {} is currently not supported for ONNX conversion".format(func.type))
        if callable(op_type):
            return op_type(func)

        variables = self._net.variable
        input_types = self._input_types
        output_types = self._output_types
        broadcast_target = self._broadcast_target

        n = onnx.helper.make_node(
            op_type,
            func.input,
            func.output,
            name=fork_name(op_type))
        nl = []
        if func.type == "GlobalAveragePooling":
            # We wipeout the node name to avoid a bug?
            # that occurs when we use a GlobalAveragePooling node with a name
            # "Conv" or "Pool" contained.
            # Caffe2 issue is here:
            # https://github.com/caffe2/caffe2/issues/1971
            # Because a GlobalAveragePooling operator does not contain a kernel, we get an error at the
            # following code if we have a specific name.
            # https://github.com/caffe2/caffe2/blob/master/caffe2/operators/conv_pool_op_base.h#L167
            # The above caffe2 code should be checking the node's operator name and not the node's name.
            n.name = ""
            nl.append(n)
        elif func.type == "Reshape":
            # Convert Reshape size to a constant
            rp = func.reshape_param
            x = func.input[0]
            c_out = x + "_shape"
            c = generate_constant(c_out, func.name + "_shape",
                                  TensorProto.INT64, [len(rp.shape.dim)],
                                  rp.shape.dim)
            nl.append(c)
            # Add resize target shape as the second input
            del n.input[:]
            n.input.extend([x, c_out])
            nl.append(n)
        elif func.type == "Transpose":
            tp = func.transpose_param
            p = onnx.helper.make_attribute("perm", tp.axes)
            n.attribute.extend([p])
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
            set_reduction_attrs(n, sp)
            nl.append(n)
        elif func.type == "Mean":
            mp = func.mean_param
            set_reduction_attrs(n, mp)
            nl.append(n)
        elif func.type == "Max":
            mp = func.max_param
            set_reduction_attrs(n, mp)
            nl.append(n)
        elif func.type == "Min":
            mp = func.min_param
            set_reduction_attrs(n, mp)
            nl.append(n)
        elif func.type == "Prod":
            pp = func.prod_param
            set_reduction_attrs(n, pp)
            nl.append(n)
        elif func.type == "BroadcastTo":
            # BroadcastTo conversion only works when the
            # broadcasted buffer is used as second input for the following:
            # Add, And, Div, Equal, Greater,
            # Less, Mul, Or, Pow, Sub, Xor
            bp = func.broadcast_to_param
            broadcast_target[func.output[0]] = (func.input[1], bp.axis)
            # we do not append node here because BroadcastTo should disappear
        elif func.type == "Constant":
            cp = func.constant_param
            shape = list(self._var_dict[func.output[0]].dim[:])
            val = [cp.val]*np.prod(shape)
            t = onnx.helper.make_tensor("Constant",
                                        data_type=TensorProto.FLOAT,
                                        dims=shape, vals=val)
            p = onnx.helper.make_attribute("value", t)
            n.attribute.extend([p])
            nl.append(n)
        elif func.type == "IsNaN":
            self._output_types[func.output[0]] = TensorProto.BOOL
            nl.append(n)
        elif func.type == "Where":
            self._input_types[func.input[0]] = TensorProto.BOOL
            nl.append(n)
        elif func.type == "IsInf":
            self._output_types[func.output[0]] = TensorProto.BOOL
            nl.append(n)
        else:
            # Simply append node to list
            nl.append(n)
        return nl

    def create_graph(self):
        net = self.set_network()
        self.set_shape_all()
        self._model_proto.graph.name = net.name
        for f in net.function:
            nl = self.set_nodes(f)
            self._model_proto.graph.node.extend(nl)

        if len(self._broadcast_target) > 0:
            # If a broadcast target buffer is not used for any of the supported
            # operator's inputs, we throw an error.
            raise ValueError("BroadcastTo targets must be used in conjunction"
                             " with certain operators in order to get converted to ONNX")
        self.set_variables()

    def create_model(self):
        mp = ModelProto()
        mp.ir_version = ONNX_IR_VERSION
        op = mp.opset_import.add()
        op.domain = ""  # empty string indicates ONNX domain
        op.version = self._opset
        # nn_opset = mp.opset_import.add()
        # nn_opset.domain = NNABLA_DOMAIN
        # nn_opset.version = NNABLA_OPSET_VERSION
        mp.producer_name = PRODUCER_NAME
        mp.producer_version = PRODUCER_VERSION
        mp.domain = NNABLA_DOMAIN
        self._model_proto = mp

    def dump_nnp(self, fn):
        import os
        keyname = os.path.splitext(fn)[0]
        nnp_fn = keyname + '.nnp.dump'
        with open(nnp_fn, "w") as f:
            f.write(str(self._nnp))
        print('{} is written.'.format(nnp_fn))

    def dump_onnx(self, fn):
        import os
        keyname = os.path.splitext(fn)[0]
        onnx_dump = keyname + '.onnx.dump'
        with open(onnx_dump, "w") as f:
            f.write(str(self._model_proto))
        print('{} is written.'.format(onnx_dump))

    def dump_graph(self):
        in_d = {}
        init_d = {}
        for input in self._model_proto.graph.input:
            in_d[input.name] = [
                d.dim_value for d in input.type.tensor_type.shape.dim]
        for init in self._model_proto.graph.initializer:
            init_d[init.name] = [d for d in init.dims]
        for node in self._model_proto.graph.node:
            for i in node.input:
                if i in in_d:
                    if i in init_d:
                        print("{} : {} <- {}".format(i, in_d[i], init_d[i]))
                    else:
                        print("{} : {}".format(i, in_d[i]))
            print(node)

    def export_model_proto(self):
        self.create_model()
        self.create_graph()
        return self._model_proto

    def execute(self, file_path):
        # if debug, please uncomment it.
        # self.dump_nnp(file_path)

        self.create_model()
        self.create_graph()
        with open(file_path, "wb") as f:
            f.write(self._model_proto.SerializeToString())

        # if debug, please uncomment it.
        # self.dump_onnx(file_path)
        # self.dump_graph()
