# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
