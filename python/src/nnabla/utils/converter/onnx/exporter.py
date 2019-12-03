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

from nnabla.utils import nnabla_pb2
from functools import partial
import numpy as np
import re
try:
    import onnx
    from .utils import *
    from onnx import (ModelProto, TensorProto, TensorShapeProto)
except:
    print('ONNX export support disabled because onnx python package is not found.')
    print(' You may install onnx package with "pip install onnx".')

TENSOR_TYPE_TO_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.BOOL: np.bool,
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

    i = graph.input.add()
    i.name = param_name
    i.type.tensor_type.elem_type = dtype
    dims = [create_dim(d) for d in shape]
    i.type.tensor_type.shape.dim.extend(dims)


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
    def __init__(self, nnp, batch_size, opset="7"):
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
        self._opset = int(re.sub("\D", "", opset))
        # Dictionary used to convert NNabla function names to ONNX op_type

        # opset_6 default op table
        table_op_set_6 = {
            "Dropout": partial(self.Dropout, "6"),
            "BatchNormalization": partial(self.BatchNormalization, "6"),
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
            "Pad": "Pad",
            "ReLU": "Relu",
            "PReLU": self.PReLU,
            "LeakyReLU": "LeakyRelu",
            "Concatenate": self.Concatenate,
            "GlobalAveragePooling": "GlobalAveragePool",
            "MaxPooling": partial(self.BasePooling, 'MaxPool'),
            "AveragePooling": partial(self.BasePooling, 'AveragePool'),
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
            "Split": self.Split,
            "Stack": self.Stack,
            "Slice": self.Slice_6,
            "Deconvolution": partial(self.BaseDeconvolution, 'Deconvolution'),
            "Flip": self.Flip,
            "OneHot": self.OneHot,
            "Unpooling": self.Unpooling_6,
            "DepthwiseConvolution": partial(self.BaseConvolution, 'DepthwiseConvolution'),
            "BinaryConnectConvolution": partial(self.BaseConvolution, 'BinaryConnectConvolution'),
            "Convolution": partial(self.BaseConvolution, 'Convolution'),
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
            "ReLU6": self.ReLU6,
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
        }

        table_op_set_7 = {
            "Dropout": partial(self.Dropout, "7"),
            "BatchNormalization": partial(self.BatchNormalization, "7"),
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
            "SumPooling": self.SumPooling,
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
        }
        table_op_set_9 = dict(table_op_set_7, **table_op_set_9)

        # opset_ support for SNPE
        table_op_set_9_x = {
            "Affine": partial(self.BaseAffine, "Affine", '9x')
        }
        table_op_set_9_x = dict(table_op_set_9, **table_op_set_9_x)

        opver_impl_map = {
            "6": table_op_set_6,
            "7": table_op_set_7,
            "9": table_op_set_9,
            "9x": table_op_set_9_x
        }
        try:
            opset = int(opset)
            if opset <= 6:
                self.nnabla_function_type_to_onnx_optype = opver_impl_map.get(
                    "6")
            else:
                self.nnabla_function_type_to_onnx_optype = opver_impl_map.get(
                    str(opset), table_op_set_9_x)
        except:
            self.nnabla_function_type_to_onnx_optype = opver_impl_map.get(
                opset, table_op_set_9_x)

    def Dropout(self, opset, func):
        # Since only export executor network from nnp,
        # Dropout works only for test.
        # we set Dropout to is test mode
        n = onnx.helper.make_node(
            'Dropout',
            func.input,
            func.output,
            name=func.name
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
                    name=func.name
                )
                nl.append(n)
            else:  # opset >= 7
                input0_shape_len = len(self._var_dict[func.input[0]].dim)
                input1_shape_len = len(self._var_dict[second_input].dim)
                unsqueeze_output = fork_name("broadcast_unsqueeze")
                trailing = list(
                    range(input1_shape_len+1, input0_shape_len))
                unsqueeze = onnx.helper.make_node(
                    'Unsqueeze',
                    [second_input],
                    [unsqueeze_output],
                    axes=list(range(axis)) + trailing,
                    name="broadcast_unsqueeze"
                )
                nl.append(unsqueeze)
                n = onnx.helper.make_node(
                    func_name,
                    [func.input[0], unsqueeze_output],
                    func.output,
                    name=func.name)
                nl.append(n)
            if func_name == "And" or func_name == "Or" or func_name == "Xor":
                self._input_types[second_input] = TensorProto.BOOL
            del self._broadcast_target[bt]
        else:
            n = onnx.helper.make_node(
                func_name,
                func.input,
                func.output,
                name=func.name)
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

    def BaseConvolution(self, func_name, func):
        nl = []
        input_x_shape = np.array(
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
            group = input_x_shape[cp.base_axis]
            weight_shape.insert(1, 1)
            proto_weight_shape = self._var_dict[inputs[1]]
            del proto_weight_shape.dim[:]
            proto_weight_shape.dim.extend(weight_shape)
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
            input_x_shape = np.array(np.concatenate(
                ([np.prod(input_x_shape[:cp.base_axis])], input_x_shape[cp.base_axis:], [1])))

            input_w_shape = np.array(
                [d for d in self._var_dict[func.input[1]].dim] + [1])
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(input_w_shape)
        elif len(pads) > 1:  # N-D convolution:
            input_x_shape = np.array(np.concatenate(
                ([np.prod(input_x_shape[:cp.base_axis])], input_x_shape[cp.base_axis:])))

        if cp.base_axis > 1:
            # Reshape input[0]
            output_x_reshape_name = fork_name("output_x_reshape")
            n = generate_reshape(self._model_proto.graph, func.input[0], output_x_reshape_name,
                                 input_x_shape)
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
        input_x_shape = np.array(
            [d for d in self._var_dict[func.input[0]].dim])
        output_y_shape = np.array(
            [d for d in self._var_dict[func.output[0]].dim])
        if len(pads) == 1:  # 1-D convolution
            # Convert 1-D to 2-D for snpe
            kernel_shape += [1]
            dilations += [1]
            strides += [1]
            pads += [0]
            input_x_shape = np.array(np.concatenate(
                ([np.prod(input_x_shape[:cp.base_axis])], input_x_shape[cp.base_axis:], [1])))

            # Reshape input[1]
            input_w_shape = np.array(
                [d for d in self._var_dict[func.input[1]].dim] + [1])
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(input_w_shape)
        elif len(pads) > 1:  # N-D convolution:
            input_x_shape = np.array(np.concatenate(
                ([np.prod(input_x_shape[:cp.base_axis])], input_x_shape[cp.base_axis:])))

        if cp.base_axis > 1:
            # Reshape input[0]
            output_x_reshape_name = fork_name("output_x_reshape")
            n = generate_reshape(self._model_proto.graph, func.input[0], output_x_reshape_name,
                                 input_x_shape)
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

        if cp.base_axis > 1:
            n = generate_reshape(self._model_proto.graph, output, func.output[0],
                                 output_y_shape)
            nl.append(n)
        else:
            nl[-1].output[0] = func.output[0]

        return nl

    def BasePooling(self, onnx_func, func):
        nl = []
        input = func.input[0]
        output = func.output[0]
        input_shape = self._var_dict[input].dim
        output_shape = self._var_dict[func.output[0]].dim
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
            rout = input + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout
            output = func.output[0] + "_reshape"

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
            pad_out = input + "_pad"
            n = onnx.helper.make_node(
                'Pad',
                [input],
                [pad_out],
                mode=pad_mode,
                value=value,
                pads=pads
            )
            input = pad_out
            nl.append(n)

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

    def BatchNormalization(self, opset, func):
        nl = []
        bnp = func.batch_normalization_param
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
            bn_input = [inputs[i] for i in onnx_order[:3]]
            n = onnx.helper.make_node(
                'InstanceNormalization',
                bn_input,
                [outputs[0]],
                epsilon=1e-5
            )
        else:
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
                # spatial=1 # say: "Don't know map unexpected argument spatial."
                # different from SPEC.
            )
            if opset == "6":
                b = onnx.helper.make_attribute("is_test", 1)
                n.attribute.extend([b])
        nl.append(n)

        if input_shape_reshape != input_shape:
            output_y_shape = np.array(
                [d for d in self._var_dict[func.output[0]].dim])
            n = generate_reshape(self._model_proto.graph, outputs[0], func.output[0],
                                 output_y_shape)
            nl.append(n)
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
            rout = input + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout
            output = func.output[0] + "_reshape"
        scales = list(
            map(lambda f: float(f), [1.0, 1.0] + func.unpooling_param.kernel.dim[:]))
        n = onnx.helper.make_node(
            'Upsample',
            [input],
            [output],
            name=func.name,
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
            rout = input + "_reshape"
            n = generate_reshape(self._model_proto.graph, input, rout,
                                 input_shape_reshape)
            nl.append(n)
            input = rout
            output = func.output[0] + "_reshape"
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
            name=func.name
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
            name=func.name
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
        # For the case where the dilations parameter is '1', the conversion, inference, compare results is no problem.
        # If the dilations parameter is not '1', the conversion result is OK, But the inference results have problems,
        # the inference tools currently in use are CNTK, Caffe2, ONNXRuntime:
        # CNTK
        # Currently, CNTK does not support 'ConvTranspose'.
        #
        # Caffe2
        # Caffe2 seems not support the case that dilations not equal to '1'. According to investigation, when the input
        # shapes are same, dilations are different, Caffe2 output same infer shape, which is not expected. We supposed
        # Caffe2 did not support dilations != 1.
        #
        # ONNXRuntime
        # The formula used to infer output shape is different from NNabla.
        # ONNXRuntime calculates the output space size for 'ConvTranspose' as :
        # L_i = (Li - 1) Si + Ki + Di - 1 + adj - padhead - padtail
        # Where Li is the spatial size, Si is the stride, Ki is the kernel size, Di is the dilation,
        # adj is output_padding, padhead is the pad start, padtail is the pad end.
        #
        # NNabla is using the formula as :
        # L_i = (Li - 1) Si - 2Pi + Di (Ki - 1) + 1
        # Where Si is the stride, Li is the spatial size, Pi is the padding, Di is the dilation,
        # and Ki is the kernel size for i-th spatial dimension.
        # If the dilations parameter is not '1', the output shape is different.
        nl = []
        inputs = func.input[:]
        input_x_shape = np.array(
            [d for d in self._var_dict[func.input[0]].dim])
        output_y_shape = np.array(
            [d for d in self._var_dict[func.output[0]].dim])
        weight_shape = [d for d in self._var_dict[func.input[1]].dim]
        if func_name == "Deconvolution":
            dp = func.deconvolution_param
            group = dp.group
        elif func_name == "DepthwiseDeconvolution":
            dp = func.depthwise_deconvolution_param
            group = output_y_shape[dp.base_axis]
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
            input_x_shape = np.array(np.concatenate(
                ([np.prod(input_x_shape[:dp.base_axis])], input_x_shape[dp.base_axis:], [1])))

            weight_shape += [1]
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(weight_shape)
        elif len(pads) > 1:  # N-D Deconvolution
            input_x_shape = np.array(np.concatenate(
                ([np.prod(input_x_shape[:dp.base_axis])], input_x_shape[dp.base_axis:])))

        if dp.base_axis > 1:
            # Reshape input[0]
            output_x_reshape = fork_name("output_x_reshape")
            n = generate_reshape(self._model_proto.graph, func.input[0],
                                 output_x_reshape, input_x_shape)
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
                group=group
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
                group=group
            )
            nl.append(n)

        return nl

    def _elem_op(self, func, op_name, val):
        # Todo: how to exploit broadcasting feature to shrink
        #       the size of val is a topic remained to the
        #       future.
        v = np.ones(self._var_dict[func.input[0]].dim) * val
        param_name = fork_name(func.input[0])
        init = self._model_proto.graph.initializer.add()
        init.name = param_name
        init.data_type = TensorProto.FLOAT
        init.dims.extend(list(v.shape))
        init.raw_data = v.astype(np.float32).tostring()

        i = self._model_proto.graph.input.add()
        i.name = param_name
        i.type.tensor_type.elem_type = TensorProto.FLOAT
        dims = [create_dim(d) for d in v.shape]
        i.type.tensor_type.shape.dim.extend(dims)
        inputs = [i for i in func.input]
        n = onnx.helper.make_node(
            op_name,
            [param_name] + inputs,
            func.output,
            name=func.name
        )
        return [n]

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

    def Slice_6(self, func):
        """
        nnabla slice assume a batch dimension existed in
        the shape of input data.
        Onnx caffe2 implementation only support one dimension
        for each slice, hence, we connect multiple slice
        node to implement slice with multiple axis
        """
        s0 = [d for d in func.slice_param.start]
        e0 = [d for d in func.slice_param.stop]
        s1 = [0] * len(self._var_dict[func.input[0]].dim)
        e1 = [d for d in self._var_dict[func.input[0]].dim]
        if len(s1) - len(s0) and len(s1) - len(e0):
            s0 = [0] + s0
            e0 = [self._batch_size] + e0
        nl = []
        steps = [d for d in func.slice_param.step]
        for i in steps:
            if i != 1:
                raise ValueError('Currently, step != 1 not supported!')
        for i, (m, n, s, e) in enumerate(zip(s0, e0, s1, e1)):
            if m > s or n < e:
                starts = s1[:]
                ends = e1[:]
                starts[i] = m
                ends[i] = n
                n = onnx.helper.make_node(
                    "Slice",
                    func.input,
                    func.output,
                    starts=starts,
                    ends=ends,
                )
                nl.append(n)

        if not nl:
            # full equal, no slice
            # we have to create a node
            # to pass the data
            starts = s1[:]
            ends = e1[:]
            n = onnx.helper.make_node(
                "Slice",
                func.input,
                func.output,
                starts=starts,
                ends=ends,
            )
            nl.append(n)

        for i in range(len(nl)-1):
            fork = fork_name("SliceIter")
            del nl[i].output[:]
            nl[i].output.extend([fork])
            del nl[i+1].input[:]
            nl[i+1].input.extend([fork])

        return nl

    def Slice_9(self, func):
        """
        ONNXRuntime implement slice with multiple axis.
        """
        s0 = [d for d in func.slice_param.start]
        e0 = [d for d in func.slice_param.stop]
        s1 = [0] * len(self._var_dict[func.input[0]].dim)
        e1 = [d for d in self._var_dict[func.input[0]].dim]
        if len(s1) - len(s0) and len(s1) - len(e0):
            s0 = [0] + s0
            e0 = [self._batch_size] + e0
        nl = []

        starts = s1[:]
        ends = e1[:]
        steps = [d for d in func.slice_param.step]
        for i in steps:
            if i != 1:
                raise ValueError('Currently, step != 1 not supported!')
        for i, (m, n, s, e) in enumerate(zip(s0, e0, s1, e1)):
            if m > s:
                starts[i] = m
            if n < e:
                ends[i] = n
        n = onnx.helper.make_node(
            "Slice",
            func.input,
            func.output,
            name=func.name,
            starts=starts,
            ends=ends,
        )
        nl.append(n)

        return nl

    def Stack(self, func):
        nl = []
        outputs = []
        for i, x in enumerate(func.input):
            output_name = fork_name(x)
            n = onnx.helper.make_node(
                "Unsqueeze",
                [x],
                [output_name],
                name="Unsqueeze_Stack_{}".format(i))
            attr = onnx.helper.make_attribute("axes", [func.stack_param.axis])
            n.attribute.extend([attr])
            nl.append(n)
            outputs.append(output_name)
        n = onnx.helper.make_node(
            "Concat",
            outputs,
            func.output,
            name="Concat_Stack")
        attr = onnx.helper.make_attribute("axis", func.stack_param.axis)
        n.attribute.extend([attr])
        nl.append(n)
        return nl

    def Split(self, func):
        nl = []
        outputs = [fork_name(out) for out in func.output]

        n = onnx.helper.make_node(
            "Split",
            func.input,
            outputs,
            name=func.name)
        attr = onnx.helper.make_attribute("axis", func.split_param.axis)
        n.attribute.extend([attr])
        nl.append(n)

        for i, x in enumerate(outputs):
            n = onnx.helper.make_node(
                "Squeeze",
                [x],
                [func.output[i]],
                name="squeeze_split_{}".format(i))
            attr = onnx.helper.make_attribute("axes", [func.split_param.axis])
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

        out_a = fork_name(inputs[0])
        base_axis = ap.base_axis

        x_shape = list(self._var_dict[inputs[0]].dim[:])
        x_shape_dims = np.array([np.prod(x_shape[:base_axis]),
                                 np.prod(x_shape[base_axis:])])
        n = generate_reshape(self._model_proto.graph, inputs[0],
                             out_a, x_shape_dims)
        nl.append(n)
        inputs[0] = out_a

        # To support SNPE, default set to `transB=1`
        if func.input[1] not in self._parameters:
            raise ValueError(
                "{} is not in network's parameters.".format(func.input[1]))

        transB = 0
        if opset == '9x':
            state = self._parameters_state.get(inputs[1], 0)
            if not state & ParameterState.TRANSPOSED:
                # make it to `transB=1`
                w_shape = list(self._var_dict[inputs[1]].dim[:])
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
            w_shape = list(self._var_dict[inputs[1]].dim[:])
            w_shape_dims = [w_shape[0], int(np.prod(w_shape) / w_shape[0])]
            proto_w_shape = self._var_dict[inputs[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(w_shape_dims)

        if len(inputs) <= 2:
            inputs.append(fork_name("affine_bias"))
            shape = (1, )
            raw_data = np.zeros(shape).astype(np.float32).tostring()
            add_param(self._model_proto.graph,
                      inputs[2], TensorProto.FLOAT, shape, raw_data)
        else:
            bias_shape = list(self._var_dict[inputs[2]].dim[:])
            new_bias_shape = [np.prod(bias_shape)]
            proto_bias_shape = nnabla_pb2.Shape()
            proto_bias_shape.dim.extend(new_bias_shape)
            self._var_dict[inputs[2]] = proto_bias_shape

        out = fork_name(func.output[0])
        n = onnx.helper.make_node(
            "Gemm",
            inputs,
            [out],
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=transB,
            name='Gemm' + func.input[0])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        param_name = func.output[0] + '_shape'
        n = onnx.helper.make_node(
            "Reshape",
            [out, param_name],
            func.output,
            name='Reshape' + func.input[0])
        nl.append(n)

        output_shape = np.array(
            self._var_dict[func.output[0]].dim).astype(np.int64)
        add_param(self._model_proto.graph, param_name, TensorProto.INT64, list(
            output_shape.shape), output_shape.tostring())

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

    def SumPooling(self, func):
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
            rout = input + "_reshape"
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

        pad_out = input + "pad"
        n = onnx.helper.make_node(
            'Pad',
            [input],
            [pad_out],
            mode='constant',
            value=0.0,
            pads=pads
        )
        input = pad_out
        nl.append(n)

        apout = input + "_ap"
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

        rout = func.output[0] + "_reshape"
        output_shape = np.array(self._var_dict[func.output[0]].dim)
        n = generate_reshape(self._model_proto.graph, apout, rout,
                             output_shape)
        nl.append(n)

        # Counter the averaging process by multiplying kernel size
        kernel_size = np.prod(spp.kernel.dim)
        mulout = input + "_kernel"
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
        sval = x + "_scalar"
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
        dout = func.output[0] + "_div"
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

        # Reshape input_a
        input_a_shape = np.array(np.concatenate(
            ([np.prod(input_a_shape[:-2])], input_a_shape[-2:])))
        rout_a = inputs[0] + "_reshape"
        n = generate_reshape(self._model_proto.graph, inputs[0],
                             rout_a, input_a_shape)
        nl.append(n)
        inputs[0] = rout_a

        # Reshape input_b
        input_b_shape = np.array(np.concatenate(
            ([np.prod(input_b_shape[:-2])], input_b_shape[-2:])))
        rout_b = inputs[1] + "_reshape"
        n = generate_reshape(self._model_proto.graph, inputs[1],
                             rout_b, input_b_shape)
        nl.append(n)
        inputs[1] = rout_b

        # Transpse input_a
        if bmp.transpose_a:
            transpose_out = inputs[0] + "_transpose"
            transpose = [0, 2, 1]
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
            transpose_out = inputs[1] + "_transpose"
            transpose = [0, 2, 1]
            n = onnx.helper.make_node(
                'Transpose',
                [inputs[1]],
                [transpose_out],
                perm=transpose
            )
            nl.append(n)
            inputs[1] = transpose_out

        # MatMul
        mout = func.output[0] + "_matmul"
        n = onnx.helper.make_node(
            'MatMul',
            inputs,
            [mout]
        )
        nl.append(n)

        # Reshape
        output_shape = np.array(
            self._var_dict[func.output[0]].dim)
        n = generate_reshape(self._model_proto.graph, mout,
                             func.output[0], output_shape)
        nl.append(n)

        return nl

    def Softmax(self, opset, func):
        nl = []
        axis = func.softmax_param.axis

        # ReduceMax
        mout = func.input[0]+"_reducemax"
        n = onnx.helper.make_node(
            'ReduceMax',
            [func.input[0]],
            [mout],
            axes=[axis],
            keepdims=True
        )
        nl.append(n)

        # Sub
        sout = func.input[0]+"_sub"
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
        expout = sout+"_exp"
        n = onnx.helper.make_node(
            'Exp',
            [sout],
            [expout],
        )
        nl.append(n)

        # ReduceSum
        sumout = expout+"_reducesum"
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
            rout = inputs[0] + "_reshape"
            n = generate_reshape(self._model_proto.graph, inputs[0],
                                 rout, input0_shape_reshape)
            nl.append(n)
            inputs[0] = rout
            outputs[0] = func.output[0] + "_reshape"
            input_shape = list(input0_shape_reshape)

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
        dout = func.output[0] + "_sigmoid"
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
        mout = func.input[0]+"_reducemax"
        n = onnx.helper.make_node(
            'ReduceMax',
            [func.input[0]],
            [mout],
            axes=[axis],
            keepdims=True
        )
        nl.append(n)

        # Sub
        sout = func.input[0]+"_sub"
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
        expout = sout+"_exp"
        n = onnx.helper.make_node(
            'Exp',
            [sout],
            [expout],
        )
        nl.append(n)

        # ReduceSum
        sumout = expout+"_reducesum"
        n = onnx.helper.make_node(
            'ReduceSum',
            [expout],
            [sumout],
            axes=[axis],
            keepdims=True
        )
        nl.append(n)

        # Log
        logout = sumout+"_log"
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
        nout = func.output[0] + "_neg"
        n = onnx.helper.make_node("Neg",
                                  [func.input[0]],
                                  [nout])
        nl.append(n)

        # Relu
        rout0 = func.output[0] + "_relu0"
        n = onnx.helper.make_node("Relu",
                                  [func.input[0]],
                                  [rout0])
        nl.append(n)

        # Relu
        rout1 = func.output[0] + "_relu1"
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

    def ReLU6(self, func):
        # Convert Relu+Constant+Min
        nl = []
        # Relu
        rout = func.output[0] + "_relu"
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
        sout = func.output[0] + "_sigmoid"
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
        shape = func.broadcast_param.shape.dim
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
        neg_out = func.input[0]+"_neg"
        n = onnx.helper.make_node("Neg",
                                  func.input,
                                  [neg_out])
        nl.append(n)

        # Elu
        elu_out0 = func.input[0]+"_elu0"
        n = onnx.helper.make_node("Elu",
                                  func.input,
                                  [elu_out0],
                                  alpha=alpha)
        nl.append(n)

        # Elu
        elu_out1 = func.input[0]+"_elu1"
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
        pow_out = func.input[0]+"_pow"
        n = onnx.helper.make_node("Pow",
                                  [func.input[0], constant0],
                                  [pow_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Mul
        mul_out = func.input[0]+"_mul"
        n = onnx.helper.make_node("Mul",
                                  [pow_out, constant1],
                                  [mul_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Add
        add_out = func.input[0]+"_add"
        n = onnx.helper.make_node("Add",
                                  [func.input[0], mul_out],
                                  [add_out])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Sqrt
        sqrt_out = func.input[0]+"_sqrt"
        n = onnx.helper.make_node("Sqrt",
                                  [constant4],
                                  [sqrt_out])
        nl.append(n)

        # Mul
        mul_out1 = func.input[0]+"_mul1"
        n = onnx.helper.make_node("Mul",
                                  [add_out, sqrt_out],
                                  [mul_out1])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Tanh
        tanh_out = func.input[0]+"_tanh"
        n = onnx.helper.make_node("Tanh",
                                  [mul_out1],
                                  [tanh_out])
        nl.append(n)

        # Add
        add_out1 = func.input[0]+"_add1"
        n = onnx.helper.make_node("Add",
                                  [tanh_out, constant3],
                                  [add_out1])
        if opset == "6":
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
        nl.append(n)

        # Div
        div_out = func.input[0]+"_div"
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
        tanh_out = func.input[0]+"_tanh"
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
            input_reshape = func.input[0]+"_reshape"
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
        equal_out = func.output[0]+"_equal"
        n = onnx.helper.make_node("Equal",
                                  [func.input[0], c_zero_out],
                                  [equal_out])
        nl.append(n)

        # Sin
        sin_out = func.output[0]+"_sin"
        n = onnx.helper.make_node("Sin",
                                  func.input,
                                  [sin_out])
        nl.append(n)

        # Div
        div_out = func.output[0]+"_div"
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
        equal_out = func.output[0]+"_equal"
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
        less_out = func.output[0]+"_less"
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
        greater_out = func.output[0]+"_greater"
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

        equal_out = func.output[0]+'_equal'
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

        less_out = func.output[0]+'_less'
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

        greater_out = func.output[0]+'_greater'
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

                p = graph.input.add()
                p.name = param.variable_name
                p.type.tensor_type.elem_type = get_tensor_type(
                    param.variable_name, self._input_types)
                dims = [create_dim(d)
                        for d in self._var_dict[param.variable_name].dim]
                p.type.tensor_type.shape.dim.extend(dims)

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
            i = graph.input.add()
            i.name = gv.variable_name
            i.type.tensor_type.elem_type = init.data_type
            dims = [create_dim(d)
                    for d in self._var_dict[gv.variable_name].dim]
            i.type.tensor_type.shape.dim.extend(dims)

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
            name=func.name)
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
        elif func.type == "Pad":
            pp = func.pad_param
            mode_conv = {
                "constant": "constant",
                "replicate": "edge",
                "reflect": "reflect"
            }
            # separate pad values to match ONNX format
            # (S0,E0,S1,E1) => (S0,S1,E0,E1)
            dim = len(pp.pad_width) // 2
            # If we can get the dimension of the input buffer,
            # we get it here. If we cannot, we are assuming 4D input
            in_name = func.input[0]
            in_var = [v for v in variables if v.name == in_name]
            in_dim = 4
            if len(in_var) == 1 and len(in_var[0].shape.dim) > 0:
                # Found variable with valid shape.
                # If the shape dimension is zero, it means
                # that is an intermediate buffer so we can't get
                # the exact dimension at this point
                # (thus assuming 4D input).
                in_dim = len(in_var[0].shape.dim)
            elif len(in_var) > 1:
                raise ValueError("More than one buffer with"
                                 " the same buffer name found.")
            zero_dim_num = in_dim - dim
            it = iter(pp.pad_width)
            # We need to fill empty dimensions with zero padding
            # (at least this is what Caffe2 expects)
            starts = [0] * zero_dim_num
            ends = [0] * zero_dim_num
            for x in it:
                starts.append(x)
                ends.append(next(it))
            starts.extend(ends)
            pad = onnx.helper.make_attribute("pads", starts)
            m = onnx.helper.make_attribute("mode", mode_conv[pp.mode])
            if pp.mode == "constant":
                v = onnx.helper.make_attribute("value", pp.constant_value)
                n.attribute.extend([pad, m, v])
            else:
                n.attribute.extend([pad, m])
            nl.append(n)
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
