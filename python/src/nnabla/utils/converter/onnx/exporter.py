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
import nnabla.logger as logger
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
R_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
    # multiplier operators.
    del broadcast_target[target_name]
    # Return the merged input's name so we can use it if we need to
    return before_broadcast


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
    # rng = np.random.RandomState(random_seed)
    # ret = ''.join(x for x in rng.choice(list(R_CHARS), 8)) + '_{:04}'.format(random_seed)
    ret = name + '_{:04}'.format(random_seed)
    return ret


class ParameterState:
    TRANSPOSED = 1


# OnnxExporter class
class OnnxExporter:
    def __init__(self, nnp, batch_size, opset="6"):
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
            # optype with same names
            "Dropout": "Dropout",
            "Softmax": "Softmax",
            "BatchNormalization": self.BatchNormalization,
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
            "Pad": "Pad",
            # optype with different names
            "ReLU": "Relu",
            "PReLU": "PRelu",
            "LeakyReLU": "LeakyRelu",
            "Concatenate": self.Concatenate,
            "GlobalAveragePooling": "GlobalAveragePool",
            "MaxPooling": partial(self.BasePooling, 'MaxPool'),
            "AveragePooling": partial(self.BasePooling, 'AveragePool'),
            "Add2": "Add",
            "BatchMatmul": "MatMul",
            "LogicalNot": "Not",
            "ELU": "Elu",
            "SELU": "Selu",
            "Sum": "ReduceSum",
            "Mean": "ReduceMean",
            "Min": "ReduceMin",
            "Max": "ReduceMax",
            "Prod": "ReduceProd",
            "Mul2": "Mul",
            "Div2": "Div",
            "Pow2": "Pow",
            "Sub2": "Sub",
            "RSubScalar": partial(self.ScalarOperator, 'RSubScalar'),
            "RDivScalar": partial(self.ScalarOperator, 'RDivScalar'),
            "RPowScalar": partial(self.ScalarOperator, 'RPowScalar'),
            "LogicalAnd": "And",
            "LogicalOr": "Or",
            "LogicalXor": "Xor",
            "Maximum2": "Max",
            "Minimum2": "Min",
            # optype that gets converted
            "Affine": partial(self.Affine, '6'),
            "MulScalar": "Mul",
            "MinimumScalar": "Clip",
            "MaximumScalar": "Clip",
            "AddScalar": "Add",
            "PowScalar": "Pow",
            "SumPooling": "Mul",
            # optype that should get merged
            # with other operators
            "BroadcastTo": "",
            "Split": self.Split,
            "Stack": self.Stack,
            "Slice": self.Slice_6,
            "Deconvolution": self.Deconvolution,
            "Flip": self.Flip,
            "OneHot": self.OneHot,
            "Unpooling": self.Unpooling_6,
            "DepthwiseConvolution": self.DepthwiseConvolution,
            "BinaryConnectConvolution": partial(self.BaseConvolution, 'BinaryConnectConvolution'),
            "Convolution": partial(self.BaseConvolution, 'Convolution'),
            "BinarySigmoid": self.BinarySigmoid
        }

        # opset_9 table
        table_op_set_9 = {
            **table_op_set_6,
            "Affine": partial(self.Affine, '9'),
            "Slice": self.Slice_9,
            "Unpooling": self.Unpooling_9
        }

        # opset_ support for SNPE
        table_op_set_9_x = {
            **table_op_set_9,
            "Affine": partial(self.Affine, '9x')
        }

        opver_impl_map = {
            "6": table_op_set_6,
            "9": table_op_set_9,
            "9x": table_op_set_9_x
        }
        try:
            opset = int(opset)
            if opset <= 6:
                self.nnabla_function_type_to_onnx_optype = opver_impl_map.get("6")
            else:
                self.nnabla_function_type_to_onnx_optype = opver_impl_map.get(str(opset), table_op_set_9_x)
        except:
            self.nnabla_function_type_to_onnx_optype = opver_impl_map.get(opset, table_op_set_9_x)

    def _add_param(self, param_name, dtype, shape, raw_data):
        init = self._model_proto.graph.initializer.add()
        init.name = param_name
        init.data_type = dtype
        init.dims.extend(shape)
        init.raw_data = raw_data

        i = self._model_proto.graph.input.add()
        i.name = param_name
        i.type.tensor_type.elem_type = dtype
        dims = [create_dim(d) for d in shape]
        i.type.tensor_type.shape.dim.extend(dims)

    def BinarySigmoid(self, func):
        '''
        Currently, caffe2 does not support this function.
        '''
        n = onnx.helper.make_node(
            'HardSigmoid',
            func.input,
            func.output,
            alpha=1.0,
            beta=0.0
        )
        return [n]

    def _conv1d(self, weight_base, weight_shape, func, func_input, cp):
        nl = []
        kernel_shape = weight_shape[weight_base:] + [1]
        dilations = cp.dilation.dim[:] + [1]
        strides = cp.stride.dim[:] + [1]
        pads = cp.pad.dim[:] + [0]
        input_x_shape_name = fork_name("input_x_shape_name")
        input_x_shape = np.array(
            [d for d in self._var_dict[func_input[0]].dim])
        input_x_shape = np.array([np.prod(
            input_x_shape) / np.prod(input_x_shape[-2:]), input_x_shape[-2], input_x_shape[-1], 1])
        input_x_shape_shape = [len(input_x_shape)]
        self._add_param(input_x_shape_name, TensorProto.INT32,
                        input_x_shape_shape,
                        input_x_shape.astype(np.int32).tostring())
        output_x_reshape = fork_name("output_x_reshape")
        n = onnx.helper.make_node(
            'Reshape',
            [func_input[0], input_x_shape_name],
            [output_x_reshape]
        )
        nl.append(n)
        inputs = [output_x_reshape]

        input_w_shape_name = fork_name("input_w_shape_name")
        input_w_shape = np.array(
            [d for d in self._var_dict[func_input[1]].dim] + [1])
        input_w_shape_shape = [len(input_w_shape)]
        self._add_param(input_w_shape_name, TensorProto.INT32,
                        input_w_shape_shape,
                        input_w_shape.astype(np.int32).tostring())
        output_w_reshape = fork_name("output_w_reshape")
        n = onnx.helper.make_node(
            'Reshape',
            [func_input[1], input_w_shape_name],
            [output_w_reshape]
        )
        nl.append(n)
        inputs.append(output_w_reshape)

        if len(func_input) == 3:
            inputs.append(func_input[2])
        output_1d_conv = fork_name('ouput_1d_conv')
        outputs = [output_1d_conv]

        n = onnx.helper.make_node(
            'Conv',
            inputs,
            outputs,
            kernel_shape=kernel_shape,
            dilations=dilations,
            strides=strides,
            pads=pads * 2,
            group=cp.group
        )
        nl.append(n)

        output_y_shape_name = fork_name("output_y_shape_name")
        output_y_shape = np.array(
            [d for d in self._var_dict[func.output[0]].dim])
        output_y_shape_shape = [len(output_y_shape)]
        self._add_param(output_y_shape_name, TensorProto.INT32,
                        output_y_shape_shape,
                        output_y_shape.astype(np.int32).tostring())
        n = onnx.helper.make_node(
            'Reshape',
            [output_1d_conv, output_y_shape_name],
            func.output
        )
        nl.append(n)
        return nl

    def BaseConvolution(self, func_name, func):
        nl = []
        if func_name == 'Convolution':
            cp = func.convolution_param
            inputs = func.input
        elif func_name == 'BinaryConnectConvolution':
            cp = func.binary_connect_convolution_param
            inputs = [func.input[0], func.input[2]]
            if len(func.input) > 3:
                inputs += [func.input[3]]
        else:
            raise ValueError('Internal error!')
        weight_shape = [d for d in self._var_dict[func.input[1]].dim]
        weight_base = cp.base_axis + 1

        if len(cp.pad.dim[:]) == 1:  # 1-D convolution
            return self._conv1d(weight_base, weight_shape, func, inputs, cp)
        elif len(cp.pad.dim[:]) == 2:  # 2-D convolution:
            kernel_shape = weight_shape[weight_base:]
            dilations = cp.dilation.dim[:]
            strides = cp.stride.dim[:]
            pads = cp.pad.dim[:]
            input_shape = [d for d in self._var_dict[func.input[0]].dim]
            input_shape = [
                int(np.prod(input_shape) / np.prod(input_shape[-3:]))] + input_shape[-3:]
            x_shape = nnabla_pb2.Shape()
            x_shape.dim.extend(input_shape)
            self._var_dict[func.input[0]] = x_shape
        else:
            raise ValueError('N(>2)-D convolution is not supported.')

        n = onnx.helper.make_node(
            'Conv',
            inputs,
            func.output,
            kernel_shape=kernel_shape,
            dilations=dilations,
            strides=strides,
            pads=pads * 2,
            group=cp.group
        )
        nl.append(n)

        return nl

    def DepthwiseConvolution(self, func):
        cp = func.depthwise_convolution_param
        in_shape = [d for d in self._var_dict[func.input[0]].dim]
        w = [d for d in self._var_dict[func.input[1]].dim]
        out_shape = [d for d in self._var_dict[func.output[0]].dim]
        assert in_shape[cp.base_axis] * \
            cp.multiplier == out_shape[cp.base_axis]
        assert w[0] == in_shape[cp.base_axis] * cp.multiplier
        group = int(out_shape[cp.base_axis] / cp.multiplier)
        w = [int(w[0] / cp.multiplier), int(w[0] / group), w[1], w[2]]
        w_shape = nnabla_pb2.Shape()
        w_shape.dim.extend(w)
        self._var_dict[func.input[1]] = w_shape
        multiple = out_shape[cp.base_axis] / in_shape[cp.base_axis]
        assert multiple == cp.multiplier, "Invalid input/output shape!"
        n = onnx.helper.make_node(
            'Conv',
            func.input,
            func.output,
            kernel_shape=w[2:],
            dilations=cp.dilation.dim,
            strides=cp.stride.dim,
            pads=cp.pad.dim[:] * 2,
            group=group
        )
        return [n]

    def BasePooling(self, onnx_func, func):
        input_shape = self._var_dict[func.input[0]].dim
        len_input_shape = len(input_shape)
        if len_input_shape != 4:
            raise ValueError("shape({}) mismatch for onnx"
                             " {} func!".format(len_input_shape, onnx_func))

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
        else:
            raise ValueError('Internal error!')

        if ignore_border:
            pads = [d for d in pads]
            pads = [pads[0], pads[1], pads[0], pads[1]]
        else:
            subs = [kk - i % ss if i % ss != 0 else kk - ss
                    for kk, ss, i in zip(k, s, input_shape[-2:])]
            pads = [0, 0] + subs
        n = onnx.helper.make_node(
            onnx_func,
            func.input,
            func.output,
            kernel_shape=k,
            strides=s,
            pads=pads
        )
        return [n]

    def BatchNormalization(self, func):
        nl = []
        bnp = func.batch_normalization_param
        onnx_order = [0, 2, 1, 3, 4]
        if len(func.input) != len(onnx_order):
            raise ValueError(
                "The number of BatchNormalization input must be {}".format(len(onnx_order)))
        for p in func.input[1:]:
            d = sum([d if d > 1 else 0 for d in self._var_dict[p].dim])
            b_shape = nnabla_pb2.Shape()
            b_shape.dim.extend([d])
            self._var_dict[p] = b_shape
        if func.batch_normalization_param.batch_stat:
            bn_input = [func.input[i] for i in onnx_order[:3]]
            bn_output = [func.output[0]]
            n = onnx.helper.make_node(
                'InstanceNormalization',
                bn_input,
                bn_output,
                epsilon=1e-5
            )
        else:
            bn_input = [func.input[i] for i in onnx_order]
            eps = 1e-5 if bnp.eps == 0.0 else bnp.eps
            decay_rate = 0.9 if bnp.decay_rate == 0.0 \
                else bnp.decay_rate
            n = onnx.helper.make_node(
                'BatchNormalization',
                bn_input,
                [func.output[0]],
                is_test=True,
                epsilon=eps,
                momentum=decay_rate
                # spatial=1 # say: "Don't know map unexpected argument spatial."
                # different from SPEC.
            )
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

    def Unpooling_9(self, func):
        scales = list(map(lambda f: float(f), [1.0, 1.0] + func.unpooling_param.kernel.dim[:]))
        n = onnx.helper.make_node(
            'Upsample',
            func.input,
            func.output,
            name=func.name,
            scales=scales
        )
        return [n]

    def Unpooling_7(self, func):
        scales = np.array([1.0, 1.0] + func.unpooling_param.kernel.dim[:])
        scale_shape = (len(scales), )
        scale_param_name = fork_name("UpsampleScales")
        self._add_param(scale_param_name,
                        TensorProto.FLOAT,
                        scale_shape,
                        scales.astype(np.float32).tostring()
                        )
        inputs = list(func.input) + [scale_param_name]

        n = onnx.helper.make_node(
            'Upsample',
            inputs,
            func.output,
            name=func.name
        )
        return [n]


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
            self._add_param(onehot_table_name, TensorProto.FLOAT,
                            onehot_table.shape, raw_data)
            self._onehot_table[output_shape[1]] = onehot_table_name

        flatten_output = fork_name('onehotflatten')
        n = onnx.helper.make_node(
            'Flatten',
            [func.input[0]],
            [flatten_output],
            axis=0
        )
        nl.append(n)

        gather_out = fork_name('onehotgatherout')
        n = onnx.helper.make_node(
            'Gather',
            [onehot_table_name, flatten_output],
            [gather_out],
            axis=0
        )
        nl.append(n)

        shape_name = fork_name('onehotoutputshape')
        raw_data = np.array(output_shape).astype(np.int32).tostring()
        self._add_param(shape_name, TensorProto.INT32,
                        (len(output_shape),), raw_data)
        n = onnx.helper.make_node(
            'Reshape',
            [gather_out, shape_name],
            func.output
        )
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
            self._add_param(index_name, TensorProto.INT32,
                            index_shape, raw_data)
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

    def Deconvolution(self, func):
        output_name = fork_name(func.output[0])
        input_shape = self._var_dict[func.input[0]].dim
        if len(input_shape) != 4:
            raise ValueError("Currently, the input shape != 4 dims is not supported "
                             "by most of ConvTranspose function implementation.")
        kernel_shape = self._var_dict[func.input[1]].dim
        if len(kernel_shape) != 4:
            raise ValueError("Currently, the weight shape != 4 dims is not supported "
                             "by most of ConvTranspose function implementation.")
        kernel_shape = kernel_shape[2:]
        strides = func.deconvolution_param.stride.dim
        pads = func.deconvolution_param.pad.dim
        # ONNX requires (x1_b, x2_b, x1_e, x2_e) style
        pads = [pads[0], pads[1], pads[0], pads[1]]
        if func.deconvolution_param.dilation.dim != [1, 1]:
            raise ValueError("Currently, dilation != [1, 1] is not supported "
                             "by most of ConvTranspose function implementation.")
        if func.deconvolution_param.group != 1:
            raise ValueError("Currently, group != 1 is not supported "
                             "by most of ConvTranspose function implementation.")
        if len(func.input) > 2:
            b_dims = self._var_dict[func.input[2]].dim
            b_shape = nnabla_pb2.Shape()
            b_shape.dim.extend([1, b_dims[0], 1, 1])
            self._var_dict[func.input[2]] = b_shape

            node_conv_transpose = onnx.helper.make_node(
                "ConvTranspose",
                [func.input[0], func.input[1]],
                [output_name],
                pads=pads,
                strides=strides,
                kernel_shape=kernel_shape,
                name=func.name
            )

            node_add = onnx.helper.make_node(
                "Add",
                [output_name, func.input[2]],
                func.output,
                broadcast=1,
                name=func.name + "_add_bias"
            )
            return [node_conv_transpose, node_add]
        else:
            node_conv_transpose = onnx.helper.make_node(
                "ConvTranspose",
                func.input,
                func.output,
                pads=pads,
                strides=strides,
                kernel_shape=kernel_shape,
                name=func.name
            )
            return [node_conv_transpose]

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
        s0 = [0] + s0
        e0 = [d for d in func.slice_param.stop]
        e0 = [self._batch_size] + e0
        s1 = [0] * len(self._var_dict[func.input[0]].dim)
        e1 = [d for d in self._var_dict[func.input[0]].dim]
        nl = []
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
                    name=func.name,
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
                name=func.name,
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
        nnabla slice assume a batch dimension existed in
        the shape of input data.
        Onnx caffe2 implementation only support one dimension
        for each slice, hence, we connect multiple slice
        node to implement slice with multiple axis
        """
        s0 = [d for d in func.slice_param.start]
        s0 = [0] + s0
        e0 = [d for d in func.slice_param.stop]
        e0 = [self._batch_size] + e0
        s1 = [0] * len(self._var_dict[func.input[0]].dim)
        e1 = [d for d in self._var_dict[func.input[0]].dim]
        nl = []

        starts = s1[:]
        ends = e1[:]
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

    def Affine(self, opver, func):
        """
        Affine is decomposed as 3 steps:
            Reshape inputs
            Gemm
            Reshape
        """
        nl = []
        out_a = fork_name(func.input[0])
        base_axis = func.affine_param.base_axis

        x_shape = list(self._var_dict[func.input[0]].dim[:])
        x_shape_dims = [np.prod(x_shape[:base_axis]),
                        np.prod(x_shape[base_axis:])]
        x_shape_dims_name = fork_name('x_shape_dims')
        x_shape_dims_raw = np.array(x_shape_dims).astype(np.int64)
        self._add_param(x_shape_dims_name, TensorProto.INT64, list(
            x_shape_dims_raw.shape), x_shape_dims_raw.tostring())

        n = onnx.helper.make_node(
            "Reshape",
            [func.input[0], x_shape_dims_name],
            [out_a]
        )
        nl.append(n)

        # To support SNPE, default set to `transB=1`
        if func.input[1] not in self._parameters:
            raise ValueError(
                "{} is not in network's parameters.".format(func.input[1]))

        transB = 0
        if opver == '9x':
            state = self._parameters_state.get(func.input[1], 0)
            if not state & ParameterState.TRANSPOSED:
                # make it to `transB=1`
                w_shape = list(self._var_dict[func.input[1]].dim[:])
                w_shape_dims = [int(np.prod(w_shape) / w_shape[0]), w_shape[0]]
                proto_w_shape = self._var_dict[func.input[1]]
                del proto_w_shape.dim[:]
                proto_w_shape.dim.extend(w_shape_dims)
                param = self._parameters[func.input[1]]
                w_data = np.array(param.data).reshape(
                    w_shape[0], int(np.prod(w_shape) / w_shape[0]))
                d = list(np.transpose(w_data).astype(np.float32).flatten())
                del param.data[:]
                param.data.extend(d)
                self._parameters_state[func.input[1]] = state | ParameterState.TRANSPOSED
            transB = 1
        else:
            w_shape = list(self._var_dict[func.input[1]].dim[:])
            w_shape_dims = [w_shape[0], int(np.prod(w_shape) / w_shape[0])]
            proto_w_shape = self._var_dict[func.input[1]]
            del proto_w_shape.dim[:]
            proto_w_shape.dim.extend(w_shape_dims)

        if len(func.input) <= 2:
            out_c = fork_name("affine_bias")
            shape = (1, )
            raw_data = np.zeros(shape).astype(np.float32).tostring()
            self._add_param(out_c, TensorProto.FLOAT, shape, raw_data)
        else:
            bias_shape = list(self._var_dict[func.input[2]].dim[:])
            new_bias_shape = [np.prod(bias_shape)]
            proto_bias_shape = nnabla_pb2.Shape()
            proto_bias_shape.dim.extend(new_bias_shape)
            self._var_dict[func.input[2]] = proto_bias_shape
            out_c = func.input[2]

        out = fork_name(func.output[0])

        if opver == '6':
            # broadcast is needed.
            n = onnx.helper.make_node(
                "Gemm",
                [out_a, func.input[1], out_c],
                [out],
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=transB,
                broadcast=1,
                name='Gemm' + func.input[0])
        else:
            # broadcast cannot appear for opset > 6
            n = onnx.helper.make_node(
                "Gemm",
                [out_a, func.input[1], out_c],
                [out],
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=transB,
                name='Gemm' + func.input[0])
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
        self._add_param(param_name, TensorProto.INT64, list(
            output_shape.shape), output_shape.tostring())

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
                "Executor network [{}] does not found in this NNP.".format(exe.network_name))
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
        if func.type == "Dropout":
            # NNP Dropout is always is_test=false
            # since we always apply dropout when it is
            # included in a network.
            attr = onnx.helper.make_attribute("is_test", 1)
            n.attribute.extend([attr])
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
            k = onnx.helper.make_attribute("kernel_shape",
                                           weight_shape.dim[weight_base:])
            d = onnx.helper.make_attribute("dilations", cp.dilation.dim)
            s = onnx.helper.make_attribute("strides", cp.stride.dim)
            p = onnx.helper.make_attribute("pads", cp.pad.dim[:] * 2)
            g = onnx.helper.make_attribute("group", cp.group)
            n.attribute.extend([k, d, s, p, g])
            nl.append(n)
        elif func.type == "GlobalAveragePooling":
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
        elif func.type == "Softmax":
            # Softmax on NNabla does softmax ONLY along the specified axis.
            # ONNX first squashes the input dimensions to 2D based on the specified axis,
            # and then calculates the Softmax.
            # Since these two slightly differ, we show a warning here.
            logger.warning(SOFTMAX_WARNING)
            attr = onnx.helper.make_attribute("axis", func.softmax_param.axis)
            n.attribute.extend([attr])
            nl.append(n)
        elif func.type == "Reshape":
            # Convert Reshape size to a constant
            rp = func.reshape_param
            x = func.input[0]
            c_out = x + "_shape"
            c = generate_constant(c_out, func.name + "_shape",
                                  TensorProto.INT32, [len(rp.shape.dim)],
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
        elif func.type == "BatchMatmul":
            bmp = func.batch_matmul_param
            if bmp.transpose_a or bmp.transpose_b:
                raise ValueError(
                    "{} with transpose is not supported yet".format(func.type))
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
        elif (func.type == "Add2" or
              func.type == "Mul2" or
              func.type == "Div2" or
              func.type == "Pow2" or
              func.type == "Sub2"):
            # Check if the second input is a brodcast target.
            bt = func.input[1]
            if bt in broadcast_target:
                merge_broadcast(n, func, bt, broadcast_target)
            nl.append(n)
        elif (func.type == "LogicalAnd" or
              func.type == "LogicalOr" or
              func.type == "LogicalXor"):
            # Store the input/output tensor's name and convert it to boolean
            input_types[n.input[0]] = TensorProto.BOOL
            output_types[n.output[0]] = TensorProto.BOOL
            # Check if the second input is a brodcast target.
            bt = func.input[1]
            if bt in broadcast_target:
                merged = merge_broadcast(n, func, bt, broadcast_target)
                # Set the merged parameter name as BOOL
                input_types[merged] = TensorProto.BOOL
            else:
                # Set the given parameter name as BOOL
                input_types[n.input[1]] = TensorProto.BOOL
            nl.append(n)
        elif (func.type == "Less" or
              func.type == "Greater"):
            # Store the output tensor's name and convert it to boolean
            output_types[n.output[0]] = TensorProto.BOOL
            # Check if the second input is a brodcast target.
            bt = func.input[1]
            if bt in broadcast_target:
                merged = merge_broadcast(n, func, bt, broadcast_target)
            nl.append(n)
        elif func.type == "Equal":
            # Get the input data's type.
            # Since ONNX only accepts bool,int32,int64
            # while NNabla does not expose its data type,
            # we default to int64 for now.
            # TODO: Get the correct type information from NNP
            intype = TensorProto.INT64
            # Store the input/output tensor's name and convert it to boolean
            input_types[n.input[0]] = intype
            output_types[n.output[0]] = TensorProto.BOOL
            # Check if the second input is a brodcast target.
            bt = func.input[1]
            if bt in broadcast_target:
                merged = merge_broadcast(n, func, bt, broadcast_target)
                # Set the merged parameter name as BOOL
                input_types[merged] = intype
            else:
                # Set the given parameter name as BOOL
                input_types[n.input[1]] = intype
            nl.append(n)
        elif func.type == "MulScalar":
            mp = func.mul_scalar_param
            if mp.val == -1.0:
                # Convert to Neg
                n.op_type = "Neg"
            else:
                # Convert the scalar param to a Const node and add it with input
                x = func.input[0]
                sval = x + "_scalar"
                c = generate_scalar_constant(
                    sval, func.name + "_scalar", mp.val)
                del n.input[:]
                n.input.extend([x, sval])
                nl.append(c)
                # set broadcast to true
                b = onnx.helper.make_attribute("broadcast", 1)
                n.attribute.extend([b])
            nl.append(n)
        elif func.type == "MinimumScalar":
            msp = func.minimum_scalar_param
            m = onnx.helper.make_attribute("max", msp.val)
            n.attribute.extend([m])
            nl.append(n)
        elif func.type == "MaximumScalar":
            msp = func.maximum_scalar_param
            m = onnx.helper.make_attribute("min", msp.val)
            n.attribute.extend([m])
            nl.append(n)
        elif func.type == "AddScalar":
            asp = func.add_scalar_param
            # Convert the scalar param to a Const node and add it with input
            x = func.input[0]
            sval = x + "_scalar"
            c = generate_scalar_constant(sval, func.name + "_scalar", asp.val)
            nl.append(c)
            del n.input[:]
            n.input.extend([x, sval])
            # set broadcast to true
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
            nl.append(n)
        elif func.type == "PowScalar":
            psp = func.pow_scalar_param
            # Convert the scalar param to a Const node and add it with input
            x = func.input[0]
            sval = x + "_scalar"
            c = generate_scalar_constant(sval, func.name + "_scalar", psp.val)
            nl.append(c)
            del n.input[:]
            n.input.extend([x, sval])
            # set broadcast to true
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
            nl.append(n)
        elif func.type == "SumPooling":
            # SumPooling gets converted to AveragePooling+Mul.
            # Mul is used to counter the division in AveragePooling
            # since SumPooling is just summing the values in each kernel.
            # Copy kernel, stride, and pads values
            spp = func.sum_pooling_param
            if not spp.ignore_border:
                raise ValueError("SumPooling with ignore_border=False"
                                 " is not supported")
            attrs = {
                "kernel_shape": spp.kernel.dim,
                "strides": spp.stride.dim,
                "pads": spp.pad.dim[:] * 2
            }
            apin = func.input[0]
            apout = apin + "_ap"
            ap = onnx.helper.make_node("AveragePool",
                                       [apin],
                                       [apout],
                                       **attrs)
            nl.append(ap)
            # Counter the averaging process by multiplying kernel size
            kernel_size = np.prod(spp.kernel.dim)
            mulout = apin + "_kernel"
            c = generate_scalar_constant(
                mulout, func.name + "_kernel", kernel_size)
            nl.append(c)
            # Rewire Mul with average pooling output
            del n.input[:]
            n.input.extend([apout, mulout])
            # set broadcast to true
            b = onnx.helper.make_attribute("broadcast", 1)
            n.attribute.extend([b])
            nl.append(n)
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
            v = onnx.helper.make_attribute("value", pp.constant_value)
            n.attribute.extend([pad, m, v])
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
