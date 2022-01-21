# Copyright 2018,2019,2020,2021 Sony Corporation.
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

import glob
import os
import shutil
import tempfile
# TODO temporary work around to suppress FutureWarning message.
import warnings
import zipfile

import google.protobuf.text_format as text_format
import numpy as np

warnings.simplefilter('ignore', category=FutureWarning)

from nnabla.utils import nnabla_pb2
from nnabla.logger import logger


def rename_square_bracket(nnp):
    def get_renamed(str):
        ret_str = str.replace(']', '').replace('[', '_')
        if ret_str != str:
            logger.debug("{} --> {}".format(str, ret_str))
        return ret_str

    def get_renamed_list(func, field, field_name):
        fields = [get_renamed(x) for x in field]
        func.ClearField(field_name)
        field.extend(fields)

    def has_expanded():
        expanded = True
        for n in nnp.network:
            if len(n.repeat_info) > 0:
                expanded = False
        return expanded

    if not has_expanded():
        return

    # check the name of nnp.parameter
    for param in nnp.parameter:
        param.variable_name = get_renamed(param.variable_name)

    # check the name of network
    for n in nnp.network:
        for v in n.variable:
            v.name = get_renamed(v.name)
        for f in n.function:
            fields = [get_renamed(x) for x in f.input]
            f.ClearField("input")
            f.input.extend(fields)
            fields = [get_renamed(x) for x in f.output]
            f.ClearField("output")
            f.output.extend(fields)

    # check the name of optimizer
    for o in nnp.optimizer:
        for p in o.parameter_variable:
            p.variable_name = get_renamed(p.variable_name)
    # check the name of executor
    for e in nnp.executor:
        for e in e.parameter_variable:
            e.variable_name = get_renamed(e.variable_name)


class NnpExporter:
    def __init__(self, nnp, batch_size, parameter_type='protobuf', force=False):
        self._parameter_type = parameter_type
        self._force = force
        self._nnp = nnp.protobuf
        self._other_files = nnp.other_files

        # This has to be done to workaround sDeepConsolePrototype
        # weird naming rule.
        rename_square_bracket(self._nnp)

    def _write_nntxt(self, filename, nnp):
        with open(filename, 'w') as f:
            text_format.PrintMessage(nnp, f)

    def _write_protobuf(self, filename, nnp):
        with open(filename, 'wb') as f:
            f.write(nnp.SerializeToString())

    def _write_h5(self, filename, nnp):
        import h5py
        with h5py.File(filename, 'w') as hd:
            for i, param in enumerate(nnp.parameter):
                data = np.reshape(list(param.data),
                                  tuple(list(param.shape.dim)))
                dset = hd.create_dataset(
                    param.variable_name, dtype='f4', data=data)
                dset.attrs['need_grad'] = param.need_grad
                dset.attrs['index'] = i

    def _export_files(self, outdir):
        with open('{}/nnp_version.txt'.format(outdir), 'w') as f:
            f.write('0.1\n')
        if self._parameter_type == 'included':
            self._write_nntxt('{}/network.nntxt'.format(outdir), self._nnp)
        else:
            nnp_wo_parameter = nnabla_pb2.NNablaProtoBuf()
            nnp_wo_parameter.CopyFrom(self._nnp)
            nnp_wo_parameter.ClearField('parameter')
            self._write_nntxt(
                '{}/network.nntxt'.format(outdir), nnp_wo_parameter)

            if self._parameter_type == 'protobuf':
                nnp_parameter_only = nnabla_pb2.NNablaProtoBuf()
                for param in self._nnp.parameter:
                    parameter = nnp_parameter_only.parameter.add()
                    parameter.CopyFrom(param)
                self._write_protobuf(
                    '{}/parameter.protobuf'.format(outdir), nnp_parameter_only)
            elif self._parameter_type == 'h5':
                self._write_h5('{}/parameter.h5'.format(outdir), self._nnp)
            elif self._parameter_type == 'none':
                pass  # store without param.
            else:
                print('Unsupported parameter type `{}`.'.format(
                    self._parameter_type))

    def _export_nnp(self, ofile):
        try:
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(ofile, 'w') as nnpzip:
                self._export_files(tmpdir)
                for f in glob.glob('{}/*'.format(tmpdir)):
                    nnpzip.write(f, os.path.basename(f))
        finally:
            shutil.rmtree(tmpdir)

    def execute(self, *args):
        if len(args) == 1:
            if os.path.isdir(args[0]):
                self._export_files(args[0])
            else:
                ofile = args[0]
                ext = os.path.splitext(ofile)[1].lower()
                if ext == '.nnp':
                    self._export_nnp(ofile)
