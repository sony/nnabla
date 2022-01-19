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


import collections

from .refine_graph import RefineGraph
from .refine_parser import RefineParser


def _strip_node_name(name):
    if name.startswith("^"):
        return name[1:]
    else:
        return name.split(":")[0]


def find_out_terminal_node(graph_def, **kwargs):
    def add_postfix(names):
        return ["{}:0".format(n) for n in names]

    unlike_output_types = ["Const", "Assign", "NoOp", "Placeholder"]
    terminal_inputs = []
    terminal_outputs = []
    input_cnt = collections.Counter()
    need_add_postfix = kwargs.get("postfix", False)
    for node in graph_def.node:
        for input in node.input:
            input = _strip_node_name(input)
            input_cnt[input] += 1
        if node.op == 'Placeholder':
            strip_name = _strip_node_name(node.name)
            terminal_inputs.append(strip_name)

    for node in graph_def.node:
        if input_cnt[node.name] == 0 and node.op not in unlike_output_types:
            terminal_outputs.append(node.name)

    if need_add_postfix:
        terminal_inputs = add_postfix(terminal_inputs)
        terminal_outputs = add_postfix(terminal_outputs)

    return terminal_inputs, terminal_outputs


def check_optimization_criteria(nnp, batch_size):
    def find_network(nnp, exe):
        net = None
        for network in nnp.protobuf.network:
            if network.name == exe.network_name:
                net = network
        return net

    def get_input_info(exec_info, network):
        input_dict = collections.OrderedDict()
        for v in exec_info.data_variable:
            input_dict[v.variable_name] = []
        for v in network.variable:
            if v.name in input_dict:
                shape = v.shape.dim
                input_dict[v.name] = [
                    x if x > 0 else batch_size for x in shape]
        return input_dict

    state = {
        'NCHW_TO_NHWC': {
            'doc': "Convert the NCHW format to NHWC, and remove the extra nodes",
            'status': True
        }
    }
    func_list = ['Convolution', 'Deconvolution', 'MaxPooling', 'AveragePooling',
                 'SumPooling', 'Unpooling', 'Interpolate', 'RandomErase', 'MaxPoolingBackward']
    func_cnt = collections.Counter()
    exec_info = nnp.protobuf.executor[0]
    network = find_network(nnp, exec_info)
    input_dict = get_input_info(exec_info, network)
    for k, shape in input_dict.items():
        if len(shape) != 4:
            state['NCHW_TO_NHWC']['status'] = False
            break
    for func in network.function:
        if func.type in func_list:
            func_cnt[func.type] += 1
        for inp in func.input:
            if inp in input_dict and len(func.ListFields()) > 4 \
                    and hasattr(func.ListFields()[-1][1], 'base_axis') \
                    and func.ListFields()[-1][1].base_axis != 1:
                state['NCHW_TO_NHWC']['status'] = False
                break
    if len(func_cnt) == 0:
        state['NCHW_TO_NHWC']['status'] = False
    return state


class OptimizePb:
    def __init__(self, graph_def):
        self._graph_def = graph_def

    def execute(self):
        self._refine_graph = RefineGraph(self._graph_def)
        self._refine_parser = RefineParser(self._refine_graph)
        self._refine_graph.prepare()
        self._refine_parser.parse()
        return self

    def export_graph_def(self):
        return self._refine_graph.export_graph_def()

    def export_to_file(self, output_file):
        self._refine_graph.save_back(output_file)

    def get_optimization_rate(self):
        return self._refine_graph.export_optimization_rate()
