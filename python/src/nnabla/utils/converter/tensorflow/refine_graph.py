import tensorflow as tf
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
import numpy as np


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

    def _build_graph(self):
        op_dict = {}
        for node in self.graph_def.node:
            op_dict[node.name] = Operator(node)

        for node in self.graph_def.node:
            for inp in node.input:
                inp = inp.split(":")[0]
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

    def _run_tensor(self, tensor_name):
        if ':' not in tensor_name:
            tensor_name += ":0"
        tensor = self.sess.graph.get_tensor_by_name(tensor_name)
        result = self.sess.run(tensor)
        return result

    def _create_const_node(self, name, value):
        tensor_content = value.tobytes()
        dt = tf.as_dtype(value.dtype).as_datatype_enum
        tensor_shape = TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=s) for s in value.shape])
        tensor_proto = TensorProto(tensor_content=tensor_content,
                                   tensor_shape=tensor_shape,
                                   dtype=dt)
        node = tf.compat.v1.NodeDef(name=name, op='Const',
                                    attr={'value': tf.compat.v1.AttrValue(tensor=tensor_proto),
                                          'dtype': tf.compat.v1.AttrValue(type=dt)})
        return node

    def _recursive_del_node(self, op):
        for inp in op.inputs:
            self._recursive_del_node(inp)
        if not self.decrease_refcount(op):
            value = self._run_tensor(op.node.name)
            node = self._create_const_node(op.node.name, value)
            self.op_dict[op.node.name] = Operator(node)

    def _merge_run_list(self):
        for r in self.run_list:
            if r.node.op == "Split":
                num_splits = r.node.attr['num_split'].i
                if num_splits > 1:
                    self._recursive_del_node(self.op_dict[r.node.input[1]])
                    value = self._run_tensor(r.node.input[1])
                    node = self._create_const_node(r.node.input[1], value)
                    self.op_dict[r.node.input[1]] = Operator(node)
            else:
                self._recursive_del_node(self.op_dict[r.node.name])
                value = self._run_tensor(r.node.name)
                node = self._create_const_node(r.node.name, value)
                self.op_dict[r.node.name] = Operator(node)

    def _nchw_to_nhwc(self, op):
        node_name = [inp.node.name for inp in op.inputs if inp.const]
        if node_name:
            node_name = node_name[0]
            value = self._run_tensor(node_name)
            if len(value.shape) == 4:
                value = np.transpose(value, (0, 2, 3, 1))
                self._remove_node(self.op_dict[node_name])
                node = self._create_const_node(node_name, value)
                self.op_dict[node_name] = Operator(node)

    def prepare(self):
        self._build_graph()
        self._prepare_travel_graph()
        self._travel_graph(self.net_outputs)
        for r in self.run_list[:]:
            for inp in r.inputs:
                if inp in self.run_list:
                    self.run_list.remove(inp)

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

    def save_back(self, pb_file_name):
        gdef = tf.compat.v1.GraphDef()
        gdef.CopyFrom(self.graph_def)
        del gdef.node[:]
        gdef.node.extend([op.node for op in self.op_dict.values()])
        with tf.io.gfile.GFile(pb_file_name, 'wb') as f:
            f.write(gdef.SerializeToString())

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
            elif op.node.op == 'PadV2':
                paddings = self._run_tensor(op.node.input[1])
                if any(paddings.flatten().tolist()):
                    paddings[[1, 2, 3]] = paddings[[2, 3, 1]]
                    node = self._create_const_node(op.node.input[1], paddings)
                    self.op_dict[op.node.input[1]] = Operator(node)
                else:
                    self._remove_node(op)
        return op_list[-2]

    def conv_bn(self, op_list):
        conv_op = add_op = mul_value = add_value = None
        next_op_list = op_list[-1].outputs
        for op in op_list[0]:
            if op.node.op == "Conv2D":
                conv_op = op
            elif op.node.op == "Add":
                add_op = op

        for op in op_list[1:]:
            if op.node.op == "Mul":
                self._remove_node(self.op_dict[op.node.name])
                value = self._run_tensor(op.node.input[1])
                mul_value = np.transpose(value, (0, 2, 3, 1))
            elif op.node.op == "Add":
                self._remove_node(self.op_dict[op.node.name])
                value = self._run_tensor(op.node.input[1])
                add_value = np.transpose(value, (0, 2, 3, 1))

        weight_node = self.op_dict[conv_op.node.input[1].split(":")[0]]
        if weight_node.node.op == 'Const':
            weight_value = self._run_tensor(conv_op.node.input[1])
            weight_value *= mul_value
            weight_tensor = self._create_const_node(
                conv_op.node.input[1], weight_value)
            self.op_dict[conv_op.node.input[1]] = Operator(weight_tensor)
        elif weight_node.node.op == 'Split':
            weight_value = self._run_tensor(weight_node.node.input[1])
            weight_value *= mul_value
            weight_tensor = self._create_const_node(
                weight_node.node.input[1], weight_value)
            self.op_dict[weight_node.node.input[1]] = Operator(weight_tensor)
        if add_op:
            bias_value = self._run_tensor(add_op.node.input[1])
            bias_value = bias_value * mul_value + add_value
            bias_tensor = self._create_const_node(
                add_op.node.input[1], bias_value)
            self.op_dict[add_op.node.input[1]] = Operator(bias_tensor)
        else:
            bias_value = add_value
            bias_name = self.fork_name("bias")
            bias_tensor = self._create_const_node(bias_name, bias_value)
            self.op_dict[bias_name] = Operator(bias_tensor)
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
                paddings = self._run_tensor(op.node.input[1])
                if any(paddings.flatten().tolist()):
                    paddings[[1, 2, 3]] = paddings[[2, 3, 1]]
                    node = self._create_const_node(op.node.input[1], paddings)
                    self.op_dict[op.node.input[1]] = Operator(node)
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
        for op in op_list:
            if op.node.op == "Mul" or op.node.op == "Add" or op.node.op == "Sub":
                self._nchw_to_nhwc(op)
            elif op.node.op == "Mean":
                indices = np.array([1, 2], dtype=np.int32)
                indices_tensor = self._create_const_node(
                    op.node.input[1], indices)
                self.op_dict[op.node.input[1]] = Operator(indices_tensor)
        return op_list[-1]

    def affine(self, op_list):
        if not self.affine_resolved:
            transpose_perm_name = self.fork_name("transpose_perm")
            transpose_perm = np.array([0, 3, 1, 2], dtype=np.int32)
            transpose_perm_tensor = self._create_const_node(
                transpose_perm_name, transpose_perm)
            self.op_dict[transpose_perm_name] = Operator(transpose_perm_tensor)

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
                    value = self._run_tensor(op.node.input[-1])
                    if value == 1:
                        node = self._create_const_node(op.node.input[-1],
                                                       np.array(3, dtype=np.int32))
                        self.op_dict[op.node.input[-1]] = Operator(node)
                elif op.node.op == "Slice":
                    begin = self._run_tensor(op.node.input[1])
                    size = self._run_tensor(op.node.input[2])
                    if len(begin) == 4 and len(size) == 4:
                        begin[[1, 2, 3]] = begin[[2, 3, 1]]
                        size[[1, 2, 3]] = size[[2, 3, 1]]
                        node = self._create_const_node(op.node.input[1], begin)
                        self.op_dict[op.node.input[1]] = Operator(node)
                        node = self._create_const_node(op.node.input[2], size)
                        self.op_dict[op.node.input[2]] = Operator(node)
