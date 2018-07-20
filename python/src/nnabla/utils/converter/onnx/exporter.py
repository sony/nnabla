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
import nnabla.logger as logger
import numpy as np
try:
    import onnx
    from .utils import *
    from onnx import (ModelProto, TensorProto, TensorShapeProto)
except:
    print('ONNX export support disabled.')

TENSOR_TYPE_TO_DTYPE = {
    TensorProto.FLOAT: np.float32,
    TensorProto.BOOL: np.bool,
    TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64,
}


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
    # multipler operators.
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
            # copy all input names for scale, bias, mean, variance
            batch_norm_constants.extend(n.input[1:5])

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
    return name + '_t'


# OnnxExporter class
class OnnxExporter:
    def __init__(self, nnp, batch_size):
        self._nnp = nnp.protobuf
        self._batch_size = batch_size
        self._model_proto = None
        self._net = None
        self._var_dict = {}
        self._input_types = {}
        self._output_types = {}
        self._broadcast_target = {}
        self._executor = None
        # Dictionary used to convert NNabla function names to ONNX op_type
        self.nnabla_function_type_to_onnx_optype = {
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
            "Pad": "Pad",
            # optype with different names
            "ReLU": "Relu",
            "PReLU": "PRelu",
            "LeakyReLU": "LeakyRelu",
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
            "Min": "ReduceMin",
            "Max": "ReduceMax",
            "Prod": "ReduceProd",
            "Mul2": "Mul",
            "Div2": "Div",
            "Pow2": "Pow",
            "Sub2": "Sub",
            "LogicalAnd": "And",
            "LogicalOr": "Or",
            "LogicalXor": "Xor",
            "Maximum2": "Max",
            "Minimum2": "Min",
            "RDivScalar": "Reciprocal",
            # optype that gets converted
            "Affine": self.Affine,
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
            "Slice": self.Slice
        }

    def Slice(self, func):
        """
        nnabla slice assume a batch dimension existed in
        the shape of input data.
        """
        n = onnx.helper.make_node(
            "Slice",
            func.input,
            func.output,
            name=func.name
        )
        starts = [d for d in func.slice_param.start]
        starts = [0] + starts
        ends = [d for d in func.slice_param.stop]
        ends = [self._batch_size] + ends
        starts = onnx.helper.make_attribute("starts", starts)
        ends = onnx.helper.make_attribute("ends", ends)
        n.attribute.extend([starts, ends])
        return [n]

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

    def Affine(self, func):
        """
        Affine is decomposed as 3 steps:
            Flatten
            Gemm
            Reshape
        """
        nl = []
        out = fork_name(func.input[0])

        n = onnx.helper.make_node(
            "Flatten",
            [func.input[0]],
            [out],
            name="Flatten" + func.input[0])
        a = onnx.helper.make_attribute("axis", func.affine_param.base_axis)
        n.attribute.extend([a])
        nl.append(n)

        func.input[0] = out
        out = fork_name(func.output[0])
        n = onnx.helper.make_node(
            "Gemm",
            func.input,
            [out],
            name='Gemm' + func.input[0])
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

        output_shape = np.array(self._var_dict[func.output[0]].dim).astype(np.int64)

        init = self._model_proto.graph.initializer.add()
        init.name = param_name
        init.data_type = TensorProto.INT64
        init.dims.extend(list(output_shape.shape))
        init.raw_data = output_shape.tostring()

        i = self._model_proto.graph.input.add()
        i.name = param_name
        i.type.tensor_type.elem_type = TensorProto.INT64
        dims = [create_dim(d) for d in output_shape.shape]
        i.type.tensor_type.shape.dim.extend(dims)

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
            self._var_dict[v.name] = replace_negative_size_with_batch_size(v.shape, bs)

    def set_variables(self):
        exe = self._executor
        graph = self._model_proto.graph
        for param in self._nnp.parameter:
            init = graph.initializer.add()
            init.name = param.variable_name
            init.dims.extend(param.shape.dim)
            t = get_tensor_type(param.variable_name, self._input_types)
            init.data_type = t
            init.raw_data = np.array(
                param.data, dtype=TENSOR_TYPE_TO_DTYPE[t]).tostring()

        for iv in exe.data_variable:
            i = graph.input.add()
            i.name = iv.variable_name
            i.type.tensor_type.elem_type = get_tensor_type(
                iv.variable_name, self._input_types)
            dims = [create_dim(d) for d in self._var_dict[iv.variable_name].dim]
            i.type.tensor_type.shape.dim.extend(dims)

        for pv in exe.parameter_variable:
            p = graph.input.add()
            p.name = pv.variable_name
            p.type.tensor_type.elem_type = get_tensor_type(
                pv.variable_name, self._input_types)
            dims = [create_dim(d) for d in self._var_dict[pv.variable_name].dim]
            p.type.tensor_type.shape.dim.extend(dims)

        # Add only the final output of the graph as output
        for ov in exe.output_variable:
            o = graph.output.add()
            o.name = ov.variable_name
            o.type.tensor_type.elem_type = get_tensor_type(
                ov.variable_name, self._output_types)
            dims = [create_dim(d) for d in self._var_dict[ov.variable_name].dim]
            o.type.tensor_type.shape.dim.extend(dims)

        for gv in exe.generator_variable:
            init = graph.initializer.add()
            init.name = gv.variable_name
            init.data_type = get_tensor_type(gv.variable_name, self._input_types)
            dims = self._var_dict[gv.variable_name].dim
            init.dims.extend(dims)
            init.raw_data = generate_value(gv.type, dims, init.data_type, gv.multiplier)
            i = graph.input.add()
            i.name = gv.variable_name
            i.type.tensor_type.elem_type = init.data_type
            dims = [create_dim(d) for d in self._var_dict[pv.variable_name].dim]
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
        elif func.type == "MaxPooling":
            mpp = func.max_pooling_param
            if not mpp.ignore_border:
                raise ValueError(
                    "MaxPooling with ignore_border=False is not supported")
            # Copy kernel, stride, and pads values
            k = onnx.helper.make_attribute("kernel_shape", mpp.kernel.dim)
            s = onnx.helper.make_attribute("strides", mpp.stride.dim)
            p = onnx.helper.make_attribute("pads", mpp.pad.dim[:] * 2)
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
                raise ValueError(
                    "AveragePooling with ignore_border=False is not supported")
            # Copy kernel, stride, and pads values
            k = onnx.helper.make_attribute("kernel_shape", app.kernel.dim)
            s = onnx.helper.make_attribute("strides", app.stride.dim)
            p = onnx.helper.make_attribute("pads", app.pad.dim[:] * 2)
            n.attribute.extend([k, s, p])
            nl.append(n)
        elif func.type == "BatchNormalization":
            # We need to rearrange the input data order.
            # NNabla BatchNormalization input order: X, beta, gamma, mean, variance
            # ONNX BatchNormalization input order: X, scale, bias, mean, variance
            onnx_order = [0, 2, 1, 3, 4]
            if len(func.input) != len(onnx_order):
                raise ValueError(
                    "The number of BatchNormalization input must be {}".format(len(onnx_order)))
            onnx_input = [func.input[i] for i in onnx_order]
            del n.input[:]
            n.input.extend(onnx_input)
            bpp = func.batch_normalization_param
            if bpp.batch_stat:
                # Batch normalization for training is currently not supported
                raise ValueError(
                    "BatchNormalization with batch_stat=True is currently not supported for ONNX conversion")
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
        elif func.type == "RDivScalar":
            rp = func.r_div_scalar_param
            if rp.val != 1.0:
                raise ValueError(
                    "RDivScalar can be converted to Reciprocal only if val is 1")
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
                c = generate_scalar_constant(sval, func.name + "_scalar", mp.val)
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
            c = generate_scalar_constant(mulout, func.name + "_kernel", kernel_size)
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

        # post process of graph
        convert_parameter_shape(self._model_proto.graph)

    def create_model(self):
        mp = ModelProto()
        mp.ir_version = ONNX_IR_VERSION
        op = mp.opset_import.add()
        op.domain = ""  # empty string indicates ONNX domain
        op.version = ONNX_OPSET_VERSION
        # nn_opset = mp.opset_import.add()
        # nn_opset.domain = NNABLA_DOMAIN
        # nn_opset.version = NNABLA_OPSET_VERSION
        mp.producer_name = PRODUCER_NAME
        mp.producer_version = PRODUCER_VERSION
        mp.domain = NNABLA_DOMAIN
        self._model_proto = mp

    def dump_nnp(self, fn):
        import os
        fn = os.path.splitext(fn)[0] + '.nnp.dump'
        with open(fn, "w") as f:
            f.write(str(self._nnp))

    def execute(self, file_path):
        #self.dump_nnp(file_path)
        self.create_model()
        self.create_graph()
        with open(file_path, "wb") as f:
            f.write(self._model_proto.SerializeToString())
