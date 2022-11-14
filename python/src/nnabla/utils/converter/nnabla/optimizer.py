# Copyright 2022 Sony Group Corporation.
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

from .opt_graph import OptimizerBuilder
from .opt_parser import OptimizerParser


def optimize_nnp(graph_def):
    opt_graph = OptimizerBuilder(graph_def.default_graph())
    opt_parser = OptimizerParser(opt_graph)
    opt_graph.prepare()
    opt_parser.parse()
    return opt_graph.output()


if __name__ == '__main__':
    import nnabla as nn
    # Define when do unittest
    input_file = ''
    output_file = ''
    g = nn.graph_def.load(input_file)
    network = optimize_nnp(g)
    g.networks[network.name] = network
    g.save(output_file)
