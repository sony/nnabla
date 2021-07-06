# Copyright 2018,2019,2020,2021 Sony Corporation.
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

_SupportedInfo = collections.namedtuple(
    '_SupportedInfo', 'import_name export_name')
extensions = _SupportedInfo(import_name=['.nnp', '.onnx', '.ckpt', '.meta', '.pb', 'saved_model', '.tflite'], export_name=[
                            '.nnp', '.nnb', '.onnx', '.tflite', 'saved_model', '.pb'])
formats = _SupportedInfo(import_name=['NNP', 'ONNX', 'TF_CKPT_V1', 'TF_CKPT_V2', 'TF_PB', 'SAVED_MODEL', 'TFLITE'], export_name=[
                         'NNP', 'NNB', 'CSRC', 'ONNX', 'SAVED_MODEL', 'TFLITE', 'TF_PB'])
