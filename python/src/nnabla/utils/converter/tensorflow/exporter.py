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

from ..onnx import OnnxExporter
from onnx_tf.backend import prepare
import tensorflow as tf
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class TensorflowExporter:
    def __init__(self, nnp, batch_size, model_format='TF_PB'):
        self._nnp = nnp
        self._batch_size = batch_size
        self._model_format = model_format
        self.check_nnp_variable_name()

    def check_nnp_variable_name(self):
        def fix_variable_name(variable_name):
            if "[" in variable_name and "]" in variable_name:
                variable_name = variable_name.replace("[", "_")
                variable_name = variable_name.replace("]", "")
            if "\'" in variable_name:
                variable_name = variable_name.replace("\'", "")
            return variable_name

        network = self._nnp.protobuf.network
        executor = self._nnp.protobuf.executor
        network_name = executor[0].network_name
        parameter_variable = []
        for net in network:
            if net.name == network_name:
                for var in net.variable:
                    if var.type == 'Parameter':
                        parameter_variable.append(var.name)
                        continue
                    var.name = fix_variable_name(var.name)
                for func in net.function:
                    if func.name not in parameter_variable:
                        func.name = fix_variable_name(func.name)
                    for i, name in enumerate(func.input):
                        if name not in parameter_variable:
                            del func.input[i]
                            func.input.insert(i, fix_variable_name(name))
                    for i, name in enumerate(func.output):
                        if name not in parameter_variable:
                            del func.output[i]
                            func.output.insert(i, fix_variable_name(name))
        for var in executor[0].data_variable:
            var.variable_name = fix_variable_name(var.variable_name)
        for var in executor[0].output_variable:
            var.variable_name = fix_variable_name(var.variable_name)

    def execute(self, output):
        onnx_model = OnnxExporter(
            self._nnp, self._batch_size, opset="11").export_model_proto()
        tf_rep = prepare(onnx_model)
        if self._model_format == 'TF_PB':
            output_path = os.path.dirname(output)
            tf_model = tf_rep.tf_module.__call__.get_concrete_function(
                **tf_rep.signatures)
            frozen_func = convert_variables_to_constants_v2(
                tf_model, lower_control_flow=False)
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                              logdir=output_path,
                              name=os.path.basename(output),
                              as_text=False)
        else:
            tf_rep.export_graph(output)


class TensorflowLiteExporter:
    def __init__(self, nnp, batch_size, enable_optimize=False):
        self._nnp = nnp
        self._batch_size = batch_size
        self._enable_optimize = enable_optimize

    def execute(self, output):
        onnx_model = OnnxExporter(
            self._nnp, self._batch_size, opset="9").export_model_proto()
        saved_model = output.split(".")[0]
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(saved_model)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        with open(output, 'wb') as f:
            f.write(tflite_model)
