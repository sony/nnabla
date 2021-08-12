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

import os

import tensorflow as tf
import tflite
import tflite2onnx
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import freeze_graph
from tf2onnx import constants, tf_loader, optimizer
from tf2onnx.tfonnx import process_tf_graph

from .common import find_out_terminal_node
from ..onnx import OnnxImporter


class TensorflowImporter:
    """ Import tensorflow model to nnp model.
    """

    def __init__(self, *args, **kwargs):
        self._tf_file = args[0]
        self._tf_format = kwargs.get("tf_format")
        self._outputs = kwargs.get("outputs")
        self._inputs = kwargs.get("inputs")

    def convert_to_onnx(self, graph_def, inputs, outputs):
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name='')
        with tf_loader.tf_session(graph=tf_graph):
            g = process_tf_graph(tf_graph,
                                 continue_on_error=False,
                                 target=",".join(
                                     constants.DEFAULT_TARGET),
                                 opset=11,
                                 input_names=inputs,
                                 output_names=outputs,
                                 inputs_as_nchw=None)
        onnx_graph = optimizer.optimize_graph(g)
        model_proto = onnx_graph.make_model(
            "converted from {}".format(self._tf_file))
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
            with tf.io.gfile.GFile(self._tf_file, 'rb') as f:
                graph_def.ParseFromString(f.read())
            inputs, outputs = find_out_terminal_node(graph_def, postfix=True)
            graph_def, inputs, outputs = tf_loader.from_graphdef(
                self._tf_file, inputs, outputs)
        elif self._tf_format == 'SAVED_MODEL':
            graph_def, inputs, outputs = tf_loader.from_saved_model(
                self._tf_file, None, None)
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
                graph_def, inputs, outputs = tf_loader.from_checkpoint(
                    self._tf_file, inputs, outputs)
        onnx_model = self.convert_to_onnx(graph_def, inputs, outputs)
        onnx_importer = OnnxImporter()
        onnx_importer.import_from_onnx_model(onnx_model)
        return onnx_importer.execute()
