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

import json
import os


def optimize_nnp_model_command(input_file, output_file):
    from nnabla.utils.converter.nnabla.optimizer import optimize_nnp
    import nnabla as nn

    g = nn.graph_def.load(input_file)
    network = optimize_nnp(g)
    g.networks[network.name] = network
    g.save(output_file)


def optimize_pb_model_command(input_pb_file, output_pb_file):
    try:
        import tensorflow as tf
        from tensorflow.python.platform import gfile
        from nnabla.utils.converter.tensorflow.common import OptimizePb
    except ImportError:
        raise ImportError(
            'nnabla_converter python package is not found, install nnabla_converter package with "pip install nnabla_converter"')

    with gfile.GFile(input_pb_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        optimize = OptimizePb(graph_def).execute()
        optimize.export_to_file(output_pb_file)
        doc_file = output_pb_file.replace('.', '_') + '.json'
        with open(doc_file, 'w') as f:
            json.dump(optimize.get_optimization_rate(), f)


def optimization_command(args):
    input_file = args.input_file[0]
    output_file = args.output_file[0]
    ext = os.path.splitext(input_file)[1]
    if ext == '.pb':
        if os.path.splitext(output_file)[1] != '.pb':
            raise ValueError("Input or output file format error.")
        optimize_pb_model_command(input_file, output_file)
    elif ext == '.nnp':
        if os.path.splitext(output_file)[1] != '.nnp':
            raise ValueError("Input or output file format error.")
        optimize_nnp_model_command(input_file, output_file)
    else:
        raise ValueError(f"{ext} is unsupported file format.")


def add_optimize_command(subparsers):
    ################################################################################
    # Optimize pb model
    subparser = subparsers.add_parser(
        'optimize', help='Optimize pb model.')
    subparser.add_argument('input_file', nargs=1,
                           help='Input pre-optimized pb model.')
    subparser.add_argument('output_file', nargs=1,
                           help='Output optimized pb model.')
    subparser.set_defaults(func=optimization_command)
