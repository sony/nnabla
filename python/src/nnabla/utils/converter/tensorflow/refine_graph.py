# Copyright 2019,2020,2021 Sony Corporation.
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
import numpy as np
import tensorflow as tf
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto


class Token:
    tokens = (
        'Conv2D',
        'Split',
        'SplitV',
        'MaxPool',
        'AvgPool',
        'Add',
        'Pad',
        'Mul',
        'Identity',
        'Transpose',
        'CommonFunc',
        'Relu',
        'Abs',
        'Sub',
        'Slice',
        'ConcatV2',
        'Conv2DBackpropInput',
        'PadV2',
        'MatMul',
        'Reshape',
        'Greater',
        'Select',
        'Mean',
        'StopGradient',
        'SquaredDifference',
        'Rsqrt',
        'AddV2',
        'RealDiv',
        'Sum',
    )

    def __init__(self, op):
        if isinstance(op, str):
            self.type = op
            self.value = None
        else:
            if op.node.op in Token.tokens:
                self.type = op.node.op
                self.value = op
            else:
                self.type = 'CommonFunc'
                self.value = op
        self.lineno = 0
        self.lexpos = 0

    def __repr__(self):
        if self.value:
            return "{}:{}:{}".format(self.type,
                                     self.value.node.op, self.value.node.name)
        else:
            return "{}:{}".format(self.type, "None")


class Operator:
    def __init__(self, node):
        self.resolved = False
        self.const = False
        self.node = node
        self.inputs = []
        self.outputs = []
        self.refcount = 0


class RefineGraph:
    def __init__(self, graph_def):
        self.graph_def = graph_def
        self.sess = tf.Session()
        self.op_dict = None
        self.net_inputs = None
        self.net_outputs = None
        self.token_list = []
        self.run_list = []
        self.random_seed = 0
        self.affine_resolved = False
        self._rename_node_dict = {}
        tf.import_graph_def(self.graph_def, name='')

    def __exit__(self):
        self.sess.close()

    def fork_name(self, name):
        self.random_seed += 1
        ret = "refine_node_" + name + '_{:04}'.format(self.random_seed)
        return ret

    def input(self, param):
        pass

    def token(self):
        if self.token_list:
            tok = self.token_list.pop(0)
            return tok
        return None

    def _strip_node_name(self, name):
        if name.startswith("^"):
            return name[1:]
        else:
            return name.split(":")[0]

    def _build_graph(self):
        op_dict = {}
        for node in self.graph_def.node:
            op_dict[node.name] = Operator(node)

        for node in self.graph_def.node:
            if node.op == 'Const':
                del op_dict[node.name].node.input[:]
                continue
            for inp in node.input:
                inp = self._strip_node_name(inp)
                if inp not in op_dict:
                    raise ValueError(
                        "{} has unresolved input:{}.".format(node.op, inp))
                else:
                    op_dict[inp].refcount += 1
                    op_dict[node.name].inputs.append(op_dict[inp])
                    op_dict[inp].outputs.append(op_dict[node.name])

        inputs = set(filter(lambda op: len(op.inputs) ==
                            0, [op for op in op_dict.values()]))
        outputs = set(filter(lambda op: len(op.outputs) ==
                             0, [op for op in op_dict.values()]))

        dead_ops = inputs & outputs
        inputs -= dead_ops
        outputs -= dead_ops

        for op in dead_ops:
            del op_dict[op.node.name]

        self.net_inputs = inputs
        self.net_outputs = outputs
        self.op_dict = op_dict

    def _prepare_travel_graph(self):
        for op in self.op_dict.values():
            op.const = False
            if op.node.op in ['Const', 'Placeholder']:
                op.resolved = True
                if op.node.op == 'Const':
                    op.const = True
            else:
                op.resolved = False

    def _travel_graph(self, outputs):
        for op in outputs:
            if all([i.const for i in op.inputs]):
                self.run_list.append(op)
                op.resolved = True
                op.const = True
            elif all([i.resolved for i in op.inputs]):
                self.token_list.append(Token(op))
                op.resolved = True
            else:
                for inp in filter(lambda i: not i.resolved, op.inputs):
                    self._travel_graph([inp])
                if all([i.const for i in op.inputs]):
                    self.run_list.append(op)
                    op.const = True
                else:
                    self.token_list.append(Token(op))
                op.resolved = True

    def _run_tensor(self, tensor_name, output_num=1):
        result = []
        for i in range(output_num):
            name = tensor_name.split(":")[0] + ":" + str(i)
            tensor = self.sess.graph.get_tensor_by_name(name)
            result.append(self.sess.run(tensor))
        return result

    def _create_const_node(self, name, value):
        for i in range(len(value)):
            fixed_name = name
            if i > 0:
                original_name = name + ":" + str(i)
                fixed_name = original_name.replace(":", "_")
                self._rename_node_dict[original_name] = fixed_name
            tensor_content = value[i].tobytes()
            dt = tf.as_dtype(value[i].dtype).as_datatype_enum
            tensor_shape = TensorShapeProto(
                dim=[TensorShapeProto.Dim(size=s) for s in value[i].shape])
            tensor_proto = TensorProto(tensor_content=tensor_content,
                                       tensor_shape=tensor_shape,
                                       dtype=dt)
            node = tf.compat.v1.NodeDef(name=fixed_name, op='Const',
                                        attr={'value': tf.compat.v1.AttrValue(tensor=tensor_proto),
                                              'dtype': tf.compat.v1.AttrValue(type=dt)})
            self.op_dict[fixed_name] = Operator(node)

    def _recursive_del_node(self, op):
        for inp in op.inputs:
            self._recursive_del_node(inp)
        self.decrease_refcount(op)

    def _get_node_output_num(self, op):
        output_num = 1
        if op.node.op == "Split":
            output_num = op.node.attr['num_split'].i
        return output_num

    def _merge_run_list(self):
        for r in self.run_list:
            self._recursive_del_node(self.op_dict[r.node.name])
            output_num = self._get_node_output_num(r)
            value = self._run_tensor(r.node.name, output_num=output_num)
            self._create_const_node(r.node.name, value)

    def _nchw_to_nhwc(self, op):
        node_name = [inp.node.name for inp in op.inputs if inp.const]
        if node_name:
            node_name = node_name[0]
            value = self._run_tensor(node_name)[0]
            if len(value.shape) == 4:
                value = np.transpose(value, (0, 2, 3, 1))
                self._remove_node(self.op_dict[node_name])
                self._create_const_node(node_name, [value])

    def prepare(self):
        self._build_graph()
        self._prepare_travel_graph()
        self._travel_graph(self.net_outputs)
        for r in self.run_list[:]:
            for inp in r.inputs:
                if inp in self.run_list:
                    self.run_list.remove(inp)

            # When the number of node outputs is huge, merging will be abandoned because it takes a lot of time.
            output_num = self._get_node_output_num(r)
            if output_num > 50:
                self.run_list.remove(r)
                self.run_list.extend(r.inputs)

        self._merge_run_list()

        for name, op in self.op_dict.items():
            if op.node.op == "Placeholder":
                shape = [d.size for d in op.node.attr['shape'].shape.dim]
                if len(shape) != 4:
                    raise Exception('illegal placeholder shape')
                shape = [shape[i] for i in [0, 2, 3, 1]]
                op.node.attr['shape'].shape.Clear()
                op.node.attr['shape'].shape.CopyFrom(
                    tf.TensorShape(shape).as_proto())

            for i in range(len(op.node.input)):
                if op.node.input[i] in self._rename_node_dict:
                    op.node.input[i] = self._rename_node_dict[op.node.input[i]]

    def save_back(self, pb_file_name):
        gdef = tf.compat.v1.GraphDef()
        gdef.CopyFrom(self.graph_def)
        del gdef.node[:]
        gdef.node.extend([op.node for op in self.op_dict.values()])
        with tf.io.gfile.GFile(pb_file_name, 'wb') as f:
            f.write(gdef.SerializeToString())

    def export_graph_def(self):
        gdef = tf.compat.v1.GraphDef()
        gdef.CopyFrom(self.graph_def)
        del gdef.node[:]
        gdef.node.extend([op.node for op in self.op_dict.values()])
        return gdef

    def export_optimization_rate(self):
        before_nodes_size = len(self.graph_def.node)
        after_nodes_size = len(self.op_dict)
        info_dict = {'optimize': {'input': True}}
        doc = 'After optimization, the input of the model will be in NHWC format.\n'
        if self.affine_resolved:
            doc += 'However, the output of the model is the same as the model before optimization.' \
                ' And the inferred results can be directly compared.\n'
            info_dict['optimize']['output'] = False
        else:
            doc += 'The output of the model should also be in the NHWC format.' \
                ' Compare the inferred results of the model before optimization, it will need to transpose the inferred results.\n'
            info_dict['optimize']['output'] = True
        doc += 'The number of nodes before optimization is {}, after optimization is {}, the optimization rate is {:.2%}.'.format(
               before_nodes_size, after_nodes_size, (before_nodes_size - after_nodes_size) / before_nodes_size)

        info_dict['optimize']['doc'] = doc
        return info_dict

    def decrease_refcount(self, op):
        op.refcount -= 1
        if op.refcount == 0:
            del self.op_dict[op.node.name]
            return True
        return False

    def _remove_node(self, op):
        for prev_op in op.inputs[:]:
            if prev_op.const:
                op.inputs.remove(prev_op)
                self.decrease_refcount(prev_op)
            else:
                prev_op.outputs.remove(op)
                prev_op.outputs.extend(op.outputs)
                for next_op in op.outputs:
                    next_op.node.input[list(next_op.node.input).index(op.node.name)] = \
                        prev_op.node.name

        for next_op in op.outputs:
            next_op.inputs.remove(op)
            next_op.inputs.extend(op.inputs)

        del self.op_dict[op.node.name]

    def pool(self, op_list):
        for op in op_list:
            if op.node.op == "Transpose":
                self._remove_node(op)
            elif op.node.op == 'PadV2' or op.node.op == 'Pad':
                paddings = self._run_tensor(op.node.input[1])[0]
                if any(paddings.flatten().tolist()):
                    paddings[[1, 2, 3]] = paddings[[2, 3, 1]]
                    self._create_const_node(op.node.input[1], [paddings])
                else:
                    self._remove_node(op)
        return op_list[-2]

    def conv_bn(self, op_list):
        conv_op = add_op = mul_value = add_value = None
        next_op_list = op_list[-1].outputs
        for op in op_list[0]:
            if op.node.op == "Conv2D":
                conv_op = op
            elif op.node.op == "Add" or op.node.op == "AddV2":
                add_op = op

        for op in op_list[1:]:
            if op.node.op == "Mul":
                self._remove_node(self.op_dict[op.node.name])
                value = self._run_tensor(op.node.input[1])[0]
                mul_value = np.transpose(value, (0, 2, 3, 1))
            elif op.node.op == "Add" or op.node.op == "AddV2":
                self._remove_node(self.op_dict[op.node.name])
                value = self._run_tensor(op.node.input[1])[0]
                add_value = np.transpose(value, (0, 2, 3, 1))

        weight_value = self._run_tensor(conv_op.node.input[1])[0]
        weight_value *= mul_value
        self._create_const_node(conv_op.node.input[1], [weight_value])
        if add_op:
            bias_value = self._run_tensor(add_op.node.input[1])[0]
            bias_value = bias_value * mul_value + add_value
            self._create_const_node(add_op.node.input[1], [bias_value])
        else:
            bias_value = add_value
            bias_name = self.fork_name("bias")
            self._create_const_node(bias_name, [bias_value])
            node = tf.NodeDef()
            node.name = self.fork_name("add")
            node.op = "Add"
            node.input.extend([conv_op.node.name, bias_name])
            node.attr['T'].type = op_list[-1].node.attr['T'].type
            self.op_dict[node.name] = Operator(node)
            self.op_dict[node.name].inputs = [conv_op]
            self.op_dict[node.name].outputs = conv_op.outputs
            conv_op.outputs = [self.op_dict[node.name]]
            for op in next_op_list:
                op.node.input[list(op.node.input).index(
                    conv_op.node.name)] = node.name
                op.inputs.remove(conv_op)
                op.inputs.append(self.op_dict[node.name])
        return conv_op

    def conv2d(self, op_list):
        nodes = []
        for op in op_list:
            if op.node.op == "Pad":
                paddings = self._run_tensor(op.node.input[1])[0]
                if any(paddings.flatten().tolist()):
                    paddings[[1, 2, 3]] = paddings[[2, 3, 1]]
                    self._create_const_node(op.node.input[1], [paddings])
                else:
                    self._remove_node(op)
            elif op.node.op == "Transpose" or op.node.op == "Identity":
                self._remove_node(op)
            elif op.node.op == "Split":
                num_splits = op.node.attr['num_split'].i
                if num_splits == 1:
                    self._remove_node(op)
            elif op.node.op == "Conv2D":
                nodes.append(op)
            elif op.node.op == "Add":
                nodes.append(op)
        return nodes

    def p_relu(self, op_list):
        for op in op_list:
            if op.node.op == "Mul":
                self._nchw_to_nhwc(op)
        return op_list[-1]

    def conv_transpose(self, op_list):
        for op in op_list[:]:
            if op.node.op == "Transpose" or op.node.op == "Identity":
                self._remove_node(op)
                op_list.remove(op)
            elif op.node.op == "Split":
                num_splits = op.node.attr['num_split'].i
                if num_splits == 1:
                    self._remove_node(op)
                    op_list.remove(op)
        return op_list[0]

    def bn(self, op_list):
        if not self.affine_resolved:
            for op in op_list:
                if op.node.op == "Mul" or op.node.op == "Add" or op.node.op == "Sub" or op.node.op == "AddV2":
                    self._nchw_to_nhwc(op)
                elif op.node.op == "Mean":
                    indices = np.array([1, 2], dtype=np.int32)
                    self._create_const_node(op.node.input[1], [indices])
                elif op.node.op == "Sum":
                    indices = np.array([0, 1, 2], dtype=np.int32)
                    self._create_const_node(op.node.input[1], [indices])
                elif op.node.op == "Reshape":
                    shape = self._run_tensor(op.node.input[1])[0]
                    shape[[1, 2, 3]] = shape[[2, 3, 1]]
                    self._create_const_node(op.node.input[1], [shape])
        else:
            for op in op_list:
                if op.node.op == "Mul" or op.node.op == "Add" or op.node.op == "Sub" or op.node.op == "AddV2":
                    node_name = [
                        inp.node.name for inp in op.inputs if inp.const]
                    if node_name:
                        node_name = node_name[0]
                        value = self._run_tensor(node_name)[0]
                        if len(value.shape) == 4:
                            self._remove_node(self.op_dict[node_name])
                            self._create_const_node(node_name, [value])
        return op_list[-1]

    def affine(self, op_list):
        if not self.affine_resolved:
            transpose_perm_name = self.fork_name("transpose_perm")
            transpose_perm = np.array([0, 3, 1, 2], dtype=np.int32)
            self._create_const_node(transpose_perm_name, [transpose_perm])

            input = op_list[0].node.input[0]
            transpose_name = self.fork_name("transpose")
            node = tf.NodeDef()
            node.name = transpose_name
            node.op = "Transpose"
            node.input.extend([input, transpose_perm_name])
            node.attr['T'].type = op_list[0].node.attr['T'].type
            self.op_dict[node.name] = Operator(node)
            self.op_dict[node.name].inputs = [
                self.op_dict[input], self.op_dict[transpose_perm_name]]
            self.op_dict[node.name].outputs = op_list[0]
            self.op_dict[input].outputs.remove(op_list[0])
            self.op_dict[input].outputs.append(self.op_dict[node.name])
            op_list[0].node.input[0] = transpose_name
            op_list[0].inputs.remove(self.op_dict[input])
            op_list[0].inputs.append(self.op_dict[node.name])
            self.affine_resolved = True
        return op_list[-1]

    def binary_sigmoid(self, op_list):
        for op in op_list:
            if op.node.op == "Greater" or op.node.op == "Select":
                self._nchw_to_nhwc(op)
        return op_list[-1]

    def set_layers(self, layers):
        for op in layers:
            if not isinstance(op, list):
                if op.node.op == "ConcatV2":
                    value = self._run_tensor(op.node.input[-1])[0]
                    if value == 1:
                        self._create_const_node(
                            op.node.input[-1], [np.array(3, dtype=np.int32)])
                elif op.node.op == "Slice":
                    begin = self._run_tensor(op.node.input[1])[0]
                    size = self._run_tensor(op.node.input[2])[0]
                    if len(begin) == 4 and len(size) == 4:
                        begin[[1, 2, 3]] = begin[[2, 3, 1]]
                        size[[1, 2, 3]] = size[[2, 3, 1]]
                        self._create_const_node(op.node.input[1], [begin])
                        self._create_const_node(op.node.input[2], [size])
