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


class TensorflowExporter:
    def __init__(self, nnp, batch_size):
        self._nnp = nnp
        self._batch_size = batch_size

    def execute(self, output):
        onnx_model = OnnxExporter(
            self._nnp, self._batch_size).export_model_proto()
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(output)
