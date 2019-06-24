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
import collections
from tf2onnx import constants, loader
from tf2onnx.graph import GraphUtil
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import freeze_graph


def _strip_node_name(name):
    if name.startswith("^"):
        return name[1:]
    else:
        return name.split(":")[0]


def _find_out_terminal_node(graph_def, **kwargs):
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


class TensorflowImporter:
    """ Import tensorflow model to nnp model.
    """

    def __init__(self, *args, **kwargs):
        self._tf_file = args[0]
        self._tf_format = kwargs.get("tf_format")
        self._outputs = kwargs.get("outputs")
        self._inputs = kwargs.get("inputs")

    def convert_to_onnx(self, graph_def, inputs, outputs):

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
                                                             constants.DEFAULT_TARGET),
                                                         opset=9,
                                                         input_names=inputs,
                                                         output_names=outputs,
                                                         inputs_as_nchw=None)
        model_proto = onnx_graph.make_model(
            "converted from {}".format(self._tf_file))
        new_model_proto = GraphUtil.optimize_model_proto(model_proto)
        if new_model_proto:
            model_proto = new_model_proto
        return model_proto

    def load_checkpoint_v1(self):
        ckpt_path = os.path.dirname(self._tf_file)
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
            output_node_names=self._outputs,
            restore_op_name="",
            filename_tensor_name="",
            output_graph=None,
            clear_devices=True,
            initializer_nodes=""
        )
        return frozen_graph

    def execute(self):
        if self._tf_format == 'TF_PB':
            graph_def = graph_pb2.GraphDef()
            with tf.gfile.GFile(self._tf_file, 'rb') as f:
                graph_def.ParseFromString(f.read())
            inputs, outputs = _find_out_terminal_node(graph_def, postfix=True)
        else:
            if self._outputs is None:
                raise ImportError("Missing '--outputs' parameter.")
            if self._inputs is None:
                raise ImportError("Missing '--inputs' parameter.")

            inputs = [i + ":0" for i in self._inputs.split(",")]
            outputs = [i + ":0" for i in self._outputs.split(",")]
            if self._tf_format == 'TF_CKPT_V1':
                graph_def = self.load_checkpoint_v1()
            elif self._tf_format == 'TF_CKPT_V2':
                graph_def, inputs, outputs = loader.from_checkpoint(
                    self._tf_file, inputs, outputs)
        onnx_model = self.convert_to_onnx(graph_def, inputs, outputs)
        onnx_importer = OnnxImporter()
        onnx_importer.import_from_onnx_model(onnx_model)
        return onnx_importer.execute()
