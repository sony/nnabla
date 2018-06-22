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

import glob
import google.protobuf.text_format as text_format
import os
import shutil
import tempfile
import zipfile

# TODO temporary work around to suppress FutureWarning message.
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

from nnabla.utils import nnabla_pb2


class NnpExporter:
    def __init__(self, nnp, batch_size, parameter_type='protobuf', force=False):
        self._parameter_type = parameter_type
        self._force = force
        self._nnp = nnp.protobuf
        self._other_files = nnp.other_files

    def write_nntxt(self, filename, nnp):
        with open(filename, 'w') as f:
            text_format.PrintMessage(nnp, f)

    def write_protobuf(self, filename, nnp):
        with open(filename, 'wb') as f:
            f.write(nnp.SerializeToString())

    def write_h5(self, filename, nnp):
        import h5py
        with h5py.File(filename, 'w') as hd:
            for i, param in enumerate(nnp.parameter):
                dset = hd.create_dataset(
                    param.variable_name, dtype='f4', data=list(param.data))
                dset.attrs['need_grad'] = param.need_grad
                dset.attrs['index'] = i

    def export_files(self, outdir):
        with open('{}/nnp_version.txt'.format(outdir), 'w') as f:
            f.write('0.1\n')
        if self._parameter_type == 'included':
            self.write_nntxt('{}/network.nntxt'.format(outdir), self._nnp)
        else:
            nnp_wo_parameter = nnabla_pb2.NNablaProtoBuf()
            nnp_wo_parameter.CopyFrom(self._nnp)
            nnp_wo_parameter.ClearField('parameter')
            self.write_nntxt(
                '{}/network.nntxt'.format(outdir), nnp_wo_parameter)

            if self._parameter_type == 'protobuf':
                nnp_parameter_only = nnabla_pb2.NNablaProtoBuf()
                for param in self._nnp.parameter:
                    parameter = nnp_parameter_only.parameter.add()
                    parameter.CopyFrom(param)
                self.write_protobuf(
                    '{}/parameter.protobuf'.format(outdir), nnp_parameter_only)
            elif self._parameter_type == 'h5':
                self.write_h5('{}/parameter.h5'.format(outdir), self._nnp)
            elif self._parameter_type == 'none':
                pass  # store without param.
            else:
                print('Unsupported parameter type `{}`.'.format(
                    self._parameter_type))

    def export_nnp(self, ofile):
        try:
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(ofile, 'w') as nnpzip:
                self.export_files(tmpdir)
                for f in glob.glob('{}/*'.format(tmpdir)):
                    nnpzip.write(f, os.path.basename(f))
        finally:
            shutil.rmtree(tmpdir)

    def export(self, *args):
        if len(args) == 1:
            if os.path.isdir(args[0]):
                self.export_files(args[0])
            else:
                ofile = args[0]
                ext = os.path.splitext(ofile)[1].lower()
                if ext == '.nnp':
                    self.export_nnp(ofile)
