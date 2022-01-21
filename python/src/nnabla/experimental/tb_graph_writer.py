# Copyright 2020,2021 Sony Corporation.
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

import tensorboardX.proto.event_pb2 as event_pb2
from nnabla.parameter import get_parameters
from tensorboardX import SummaryWriter
from tensorboardX.proto.attr_value_pb2 import AttrValue
from tensorboardX.proto.graph_pb2 import GraphDef
from tensorboardX.proto.node_def_pb2 import NodeDef
from tensorboardX.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboardX.proto.types_pb2 import DT_FLOAT
from tensorboardX.proto.versions_pb2 import VersionDef


class TBGraphWriter(SummaryWriter):
    """This class is a wrapper of tensorboardX summary writer,
    which enable nn.Variable can be visualized as a graph by tensorboard.

    Prerequisite:
       Install tensorflow and tensorboardX, simply by the following commands:

       .. code-block:: plaintext

           pip install tensorflow
           pip install tensorboardX

    Please refer to the following example to use this class:

    Example:

    .. code-block:: python

        import numpy as np
        import nnabla as nn
        import nnabla.functions as F
        import nnabla.parametric_functions as PF

        def show_a_graph():
            try:
                from nnabla.experimental.tb_graph_writer import TBGraphWriter
            except Exception as e:
                print("please install tensorflow and tensorboardX at first.")
                raise e

            nn.clear_parameters()
            x = nn.Variable((2, 3, 4, 4))
            with nn.parameter_scope('c1'):
                h = PF.convolution(x, 8, (3, 3), pad=(1, 1))
                h = F.relu(PF.batch_normalization(h))
            with nn.parameter_scope('f1'):
                y = PF.affine(h, 10)

            with TBGraphWriter(log_dir='log_out') as tb:
                tb.from_variable(y, output_name="y")

    The corresponding graph can be visualized as the following:

    .. image:: ./tb_graph_writer_files/graph.PNG
       :height: 561px
       :width: 233px
       :scale: 50%

    Or, you may show scalar value or histogram of values along the increasing of iteration number
    as the following:

    .. code-block:: python

        with TBGraphWriter(log_dir='log_out') as tb:
            values = []
            for i in range(360):
                s = np.sin(i / 180.0 * np.pi)
                tb.add_scalar("show_curve/sin", s, i)
                values.append(s)

            nd_values = np.array(values)
            for i in range(10):
                tb.add_histogram("histogram", nd_values, i)
                nd_values += 0.05

    It looks like:

    .. image:: ./tb_graph_writer_files/scalar.PNG
       :height: 418px
       :width: 522px
       :scale: 50%

d
    .. image:: ./tb_graph_writer_files/histogram.PNG
       :height: 328px
       :width: 387px
       :scale: 50%


    This class writes a protobuf file to `log_dir` specified folder, thus, user should launch tensorboard
    to specify this folder:

    .. code-block:: plaintext

       tensorboard --logdir=log_out

    Then, user may check graph in a web browser, by typing the address:
       http://localhost:6006

    See Also:
        https://tensorboardx.readthedocs.io/en/latest/tensorboard.html

    """

    def __init__(self, log_dir="log_out", comment='', **kwargs):
        super(TBGraphWriter, self).__init__(log_dir, comment, **kwargs)

    def from_variable(self, leaf, output_name="output"):
        """
        Args:
          leaf (`nn.Variable`): Leaf node of graph, normally, the output variable of a network
          output_name ('str'): A given name of output variable of graph
        """
        def parse_variable(v, var_num):
            def add_variable(v, v_idx):
                v_name = parameters.get(v.data, None)
                exist = False
                if not v_name:
                    v_name, exist = get_variable_name(v, v_idx)
                if not exist:
                    shape_proto = TensorShapeProto(
                        dim=[TensorShapeProto.Dim(size=d) for d in v.shape])

                    if v.parent is None:
                        inputs = []
                    else:
                        inputs = [get_func_name(v.parent)]
                    # print("Variable: {}:{}".format(v_name, inputs))
                    nodes.append(NodeDef(
                        name=v_name.encode(encoding='utf-8'),
                        op='Variable',
                        input=inputs,
                        attr={
                            'shape': AttrValue(shape=shape_proto),
                            'dtype': AttrValue(type=DT_FLOAT)
                        }
                    ))
                return v_name

            def get_unique_variable_name(v_name_base):
                v_num = 0
                v_name = v_name_base + '_' + str(v_num)
                while v_name in unique_var_names:
                    v_num += 1
                    v_name = v_name_base + '_' + str(v_num)
                unique_var_names.add(v_name)
                return v_name

            def get_variable_name(v, v_idx):
                v_name = variables.get(v, None)
                if v_name:
                    return v_name, True
                else:
                    if v.parent is None:
                        v_name_base = "Input"
                        v_name = get_unique_variable_name(v_name_base)
                    elif not nodes:
                        v_name = output_name
                    else:
                        f_name_sections = get_func_name(v.parent).split("/")
                        f_name = f_name_sections[-1]
                        f_scope = f_name_sections[:-1]
                        base_name = "v_{}".format(f_name)
                        v_name_base = "/".join(f_scope + [base_name])
                        v_name = get_unique_variable_name(v_name_base)

                    variables[v] = v_name
                    return v_name, False

            def get_func_name(func):
                func_name = func_names.get(func, None)
                if func_name:
                    return func_name
                name_scope = loc_var['name_scope']
                for v in func.inputs:
                    v_name = parameters.get(v.data, None)
                    if v_name:
                        name_scope = '/'.join(v_name.split('/')[:-1])
                        break
                if name_scope:
                    func_name_base = '/'.join([name_scope, func.name])
                else:
                    func_name_base = func.name
                func_num = 0
                func_name = func_name_base + str(func_num)
                while func_name in unique_func_names:
                    func_num += 1
                    func_name = func_name_base + '_' + str(func_num)
                unique_func_names.add(func_name)
                func_names[func] = func_name
                return func_name

            def add_func(v):
                input_names = []
                for index, v_input in enumerate(v.parent.inputs):
                    v_name = add_variable(v_input, index)
                    input_names.append(v_name)
                # print("Function: {}:{}".format(get_func_name(v.parent), input_names))
                f_name = get_func_name(v.parent)
                if f_name in func_set:
                    return False
                attrs = []
                for k, a in v.parent.info.args.items():
                    attr = "{}={}".format(k, a)
                    attrs.append(attr)
                attr_str = ','.join(attrs).encode(encoding='utf-8')
                nodes.append(NodeDef(
                    name=f_name,
                    op=v.parent.info.type_name,
                    input=input_names,
                    attr={"parameters": AttrValue(s=attr_str)}
                ))
                func_set.add(f_name)
                return True

            name_scope = loc_var['name_scope']
            if not nodes:
                add_variable(v, var_num)
            if v.parent is None:
                add_variable(v, var_num)
            else:
                if not add_func(v):
                    return
                for idx, in_var in enumerate(v.parent.inputs):
                    name_scope_stack.append(name_scope)
                    parse_variable(in_var, idx)
                    name_scope = name_scope_stack.pop()

        nodes = []
        variables = {}
        loc_var = {}
        loc_var['name_scope'] = ''
        name_scope_stack = []
        func_names = {}
        func_set = set()
        unique_func_names = set()
        unique_var_names = set()
        parameters = {v.data: k for k,
                      v in get_parameters(grad_only=False).items()}
        parse_variable(leaf, 0)
        nodes = nodes[::-1]

        current_graph = GraphDef(node=nodes, versions=VersionDef(producer=22))
        event = event_pb2.Event(
            graph_def=current_graph.SerializeToString())
        self.file_writer.add_event(event)

    def from_graph_def(self, graph_def):
        variables = graph_def.variables
        parameters = graph_def.parameters
        functions = graph_def.functions
        inputs = graph_def.inputs
        nodes = []
        scope = {}

        for n, v in parameters.items():
            shape_proto = TensorShapeProto(
                dim=[TensorShapeProto.Dim(size=d) for d in v.shape])
            node = NodeDef(
                name=n.encode(encoding='utf-8'),
                op='Parameter',
                input=[],
                attr={
                    'shape': AttrValue(shape=shape_proto),
                    'dtype': AttrValue(type=DT_FLOAT)
                }
            )
            nodes.append(node)
            scope[n] = node

        for n, v in inputs.items():
            shape_proto = TensorShapeProto(
                dim=[TensorShapeProto.Dim(size=d) for d in v.shape])
            nodes.append(NodeDef(
                name=n.encode(encoding='utf-8'),
                op='Variable',
                input=[],
                attr={
                    'shape': AttrValue(shape=shape_proto),
                    'dtype': AttrValue(type=DT_FLOAT)
                }
            ))

        for func_name, func in functions.items():
            for o in func['outputs']:
                if o in scope:
                    node = scope[o]
                    node.input.extend([func_name])
                else:
                    if o in variables:
                        v = variables[o]
                        shape_proto = TensorShapeProto(
                            dim=[TensorShapeProto.Dim(size=d) for d in v.shape])
                        node = NodeDef(
                            name=o.encode(encoding='utf-8'),
                            op='Variable',
                            input=[func_name],
                            attr={
                                'shape': AttrValue(shape=shape_proto),
                                'dtype': AttrValue(type=DT_FLOAT)
                            }
                        )
                        nodes.append(node)
            for i in func['inputs']:
                if i in variables:
                    v = variables[i]
                    shape_proto = TensorShapeProto(
                        dim=[TensorShapeProto.Dim(size=d) for d in v.shape])
                    node = NodeDef(
                        name=o.encode(encoding='utf-8'),
                        op='Variable',
                        input=[],
                        attr={
                            'shape': AttrValue(shape=shape_proto),
                            'dtype': AttrValue(type=DT_FLOAT)
                        }
                    )
                    nodes.append(node)
                    scope[o] = node
            nodes.append(NodeDef(
                name=func_name,
                op=func['type'],
                input=func['inputs'],
                attr={"arguments": AttrValue(s='a=1'.encode(encoding='utf-8'))}
            ))

        current_graph = GraphDef(node=nodes, versions=VersionDef(producer=22))
        event = event_pb2.Event(
            graph_def=current_graph.SerializeToString())
        self.file_writer.add_event(event)
