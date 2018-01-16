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

import nnabla.utils.converter
from nnabla.utils.converter.onnx import OnnxReader, OnnxExporter

def test_conversion():
    nnabla.utils.converter.convert_files(args, args.files, output)

def test_onnx_nnp_conversion_relu():
    f = open("relu.onnx", "rb")
    r = OnnxReader(f)
    nnp = r.read()
    assert nnp is not None
    # Process nnp with naabla and get result
    # Process with caffe2 and get result
    # Compare both naabla and caffe2 results
