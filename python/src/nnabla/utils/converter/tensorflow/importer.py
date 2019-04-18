# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
from ..onnx import OnnxImporter
import tensorflow as tf
import tf2onnx
from tf2onnx.graph import GraphUtil
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import freeze_graph
# import pdb


def _strip_node_name(name):
    if name.startswith("^"):
        return name[1:]
    else:
        return name.split(":")[0]


def _find_out_terminal_node(graph_def, **kwargs):
    def add_postfix(names):
        return ["{}:0".format(n) for n in names]

    unlike_output_types = ["Const", "Assign", "Noop", "Placeholder"]
    terminal_inputs = []
    inputs = set()
    outputs = set()
    need_add_postfix = kwargs.get("postfix", False)
    for node in graph_def.node:
        strip_name = _strip_node_name(node.name)
        if node.op == 'Placeholder':
            terminal_inputs.append(_strip_node_name(node.name))
        outputs.add(strip_name)
        inputs.update(set(node.input))
    terminal_outputs = list(filter(lambda x: x not in unlike_output_types,
                                   outputs - inputs))
    if need_add_postfix:
        terminal_inputs = add_postfix(terminal_inputs)
        terminal_outputs = add_postfix(terminal_outputs)

    return terminal_inputs, terminal_outputs


class TensorflowImporter:
    """ Import tensorflow model to nnp model.
    """

    def __init__(self, *args, **kwargs):
        self._tf_file = args[0]
        self._tf_format = kwargs.get("tf_format", "TF_PB")

    def _import_from_tf_pb(self, graph_def):
        inputs, outputs = _find_out_terminal_node(graph_def, postfix=True)
        print("inputs:{}".format(inputs))
        print("outputs:{}".format(outputs))

        # FIXME: folding const = False
        graph_def = tf2onnx.tfonnx.tf_optimize(
            inputs, outputs, graph_def, False)
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=tf_graph):
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(tf_graph,
                                                         continue_on_error=False,
                                                         verbose=False,
                                                         target=",".join(
                                                             tf2onnx.tfonnx.DEFAULT_TARGET),
                                                         opset=6,
                                                         input_names=inputs,
                                                         output_names=outputs,
                                                         inputs_as_nchw=None)
        model_proto = onnx_graph.make_model("tf_model")
        new_model_proto = GraphUtil.opt_transposes_with_graph(onnx_graph,
                                                              'tf_model',
                                                              optimize=True)
        if new_model_proto:
            model_proto = new_model_proto
        return model_proto

    def import_from_tf_pb(self):
        graph_def = graph_pb2.GraphDef()
        with tf.gfile.GFile(self._tf_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        return self._import_from_tf_pb(graph_def)

    def import_from_tf_ckpt(self):
        ckpt_path = os.path.dirname(self._tf_file)
        if not ckpt_path:
            raise ValueError(
                "check point file should be in a special directory.")
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        saver = tf.train.import_meta_graph(latest_ckpt + ".meta")
        with tf.Session() as session:
            session.run(
                [
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()
                ]
            )
            saver.restore(session, latest_ckpt)
            graph_def = session.graph.as_graph_def(add_shapes=True)
        frozen_graph = freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=graph_def,
            input_saver_def=None,
            input_checkpoint=latest_ckpt,
            output_node_names="biases",
            restore_op_name="",
            filename_tensor_name="",
            output_graph=None,
            clear_devices=True,
            initializer_nodes=""
        )
        onnx_model = self._import_from_tf_pb(frozen_graph)
        return onnx_model

    def execute(self):
        if self._tf_format == 'TF_PB':
            onnx_model = self.import_from_tf_pb()
        elif self._tf_format == 'TF_CKPT':
            onnx_model = self.import_from_tf_ckpt()
        onnx_importer = OnnxImporter()
        onnx_importer.import_from_onnx_model(onnx_model)
        return onnx_importer.execute()
