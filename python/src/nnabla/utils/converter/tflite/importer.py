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

from ..onnx import OnnxImporter
import tflite
import tflite2onnx


class TFLiteImporter:
    """ Import tensorflow lite model to nnp model.
    """

    def __init__(self, file_path=''):
        self._file_path = file_path

    def execute(self):
        with open(self._file_path, 'rb') as f:
            data = f.read()
            tflite_model = tflite.Model.GetRootAsModel(data, 0)

        model = tflite2onnx.model.Model(tflite_model)
        model.convert(dict())
        onnx_importer = OnnxImporter()
        onnx_importer.import_from_onnx_model(model.onnx)
        return onnx_importer.execute()
