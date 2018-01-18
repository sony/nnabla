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

import google.protobuf.text_format as text_format
import os
import shutil
import tempfile
import zipfile

from nnabla.utils import nnabla_pb2
from onnx import (defs, checker, helper, numpy_helper, mapping,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorSetIdProto)
from onnx.helper import make_tensor, make_tensor_value_info

def onnx_model_to_protobuf(onnx_model):
    protobuf = nnabla_pb2.NNablaProtoBuf()
    # convert onnx model to nnabla protobuf
    #for ifile in self._args:
    #    print('Reading {}'.format(ifile))
    #    ext = os.path.splitext(ifile)[1].lower()
    #    if ext == '.nnp':
    #        try:
    #            tmpdir = tempfile.mkdtemp()
    #            with zipfile.ZipFile(ifile, 'r') as nnpzip:
    #                for name in nnpzip.namelist():
    #                    if os.path.splitext(name)[1].lower() in ['.nntxt', '.prototxt']:
    #                        nnpzip.extract(name, tmpdir)
    #                        with open(os.path.join(tmpdir, name), 'rt') as f:
    #                            text_format.Merge(f.read(), self._nnp)
    #                for name in nnpzip.namelist():  # Param
    #                    if os.path.splitext(name)[1].lower() in ['.protobuf', '.h5']:
    #                        nnpzip.extract(name, tmpdir)
    #                        self.load_parameters(
    #                            os.path.join(tmpdir, name))
    #        finally:
    #            shutil.rmtree(tmpdir)
    #    elif ext in ['.nntxt', '.prototxt']:
    #        with open(ifile, 'rt') as f:
    #            text_format.Merge(f.read(), self._nnp)
    #    elif ext in ['.protobuf', '.h5']:
    #        self.load_parameters(ifile)

    #if self._expand_network:
    #    self._nnp = expander.NnpExpander(self._nnp).expand()

    class nnp:
        pass
    nnp.protobuf = protobuf
    return nnp

class OnnxReader:
    def __init__(self, file_path):
        self._file_path = file_path

    #def load_parameters(self, filename):
    #    e = os.path.splitext(filename)[1].lower()
    #    if e == '.h5':
    #        import h5py
    #        with h5py.File(filename, 'r') as hd:
    #            keys = []

    #            def _get_keys(name):
    #                ds = hd[name]
    #                if not isinstance(ds, h5py.Dataset):
    #                    # Group
    #                    return
    #                # To preserve order of parameters
    #                keys.append((ds.attrs.get('index', None), name))
    #            hd.visit(_get_keys)
    #            for _, key in sorted(keys):
    #                ds = hd[key]
    #                parameter = self._nnp.parameter.add()
    #                parameter.variable_name = key
    #                parameter.shape.dim.extend(ds.shape)
    #                parameter.data.extend(ds[...].flatten())
    #                if ds.attrs['need_grad']:
    #                    parameter.need_grad = True
    #                else:
    #                    parameter.need_grad = False

    #    elif e == '.protobuf':
    #        with open(filename, 'rb') as f:
    #            self._nnp.MergeFromString(f.read())

    def read(self):
        model_proto = ModelProto()
        with open(self._file_path, "rb") as f:
            model_proto.ParseFromString(f.read())
        return onnx_model_to_protobuf(model_proto)
