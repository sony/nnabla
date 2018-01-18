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
import nnabla.utils.converter
import onnx
import onnx_caffe2.backend
import numpy as np
import nnabla as nn
from nnabla.utils.converter.onnx import OnnxReader, OnnxExporter

TEST_DATA_DIR="conversion_data"
#def test_conversion():
#    nnabla.utils.converter.convert_files(args, args.files, output)

def test_onnx_nnp_conversion_relu():
    path = os.path.join(TEST_DATA_DIR, "relu.onnx")
    # Process onnx with caffe2 backend
    model = onnx.load(path)
    c2out = onnx_caffe2.backend.run_model(model, [])
    # Process onnx with naabla
    r = OnnxReader(path)
    nnp = r.read()
    assert nnp is not None
    #nout = np.zeros((1,3,3,3))
    assert nnp.protobuf is not None
    nn.clear_parameters()
    print(nnp.protobuf)
    nn.parameter.set_parameter_from_proto(nnp.protobuf)
    param = nn.get_parameters().values()
    print(param)
    # Compare both naabla and caffe2 results
    #assert np.allclose(c2out, nout)
