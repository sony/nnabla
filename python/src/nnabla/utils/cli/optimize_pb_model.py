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

import os


def optimize_pb_model_command(args):
    try:
        import tensorflow as tf
        from tensorflow.python.platform import gfile
        from nnabla.utils.converter.tensorflow.refine_parser import RefineParser
        from nnabla.utils.converter.tensorflow.refine_graph import RefineGraph
    except:
        print("Need to install tensorflow.")

    input_pb_file = args.input_pb_file[0]
    output_pb_file = args.output_pb_file[0]
    if os.path.splitext(input_pb_file)[1] != '.pb' or \
            os.path.splitext(output_pb_file)[1] != '.pb':
        ValueError("Input or output file format error.")
    with gfile.GFile(input_pb_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        refine_graph = RefineGraph(graph_def)
        refine_parser = RefineParser(refine_graph)
        refine_graph.prepare()
        refine_parser.parse()

    refine_graph.save_back(output_pb_file)


def add_optimize_pb_model_command(subparsers):
    ################################################################################
    # Optimize pb model
    subparser = subparsers.add_parser(
        'optimize', help='Optimize pb model.')
    subparser.add_argument('input_pb_file', nargs=1,
                           help='Input pre-optimized pb model.')
    subparser.add_argument('output_pb_file', nargs=1,
                           help='Output optimized pb model.')
    subparser.set_defaults(func=optimize_pb_model_command)
