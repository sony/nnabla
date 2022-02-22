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

import os
import shutil
import tempfile
import zipfile

import google.protobuf.text_format as text_format
from nnabla.utils import nnabla_pb2

from . import expander


class NnpImporter:

    def _shrink_with_executor(self, executor):
        print(' Try to leave only executor[{}].'.format(executor.name))
        network = None
        for n in self._nnp.network:
            if n.name == executor.network_name:
                network = n
        if network is None:
            return None

        nnp = nnabla_pb2.NNablaProtoBuf()
        nnp.CopyFrom(self._nnp)

        nnp.ClearField('optimizer')
        nnp.ClearField('monitor')

        nnp.ClearField('network')
        net = nnp.network.add()
        net.CopyFrom(network)

        nnp.ClearField('executor')
        exe = nnp.executor.add()
        exe.CopyFrom(executor)

        return nnp

    def __init__(self, *args, **kwargs):
        self._args = args

        self._expand_network = False
        if 'expand_network' in kwargs:
            self._expand_network = kwargs['expand_network']

        self._executor_index = None
        if 'executor_index' in kwargs:
            self._executor_index = kwargs['executor_index']

    def load_parameters(self, filename):
        e = os.path.splitext(filename)[1].lower()
        if e == '.h5':
            import h5py
            with h5py.File(filename, 'r') as hd:
                keys = []

                def _get_keys(name):
                    ds = hd[name]
                    if not isinstance(ds, h5py.Dataset):
                        # Group
                        return
                    # To preserve order of parameters
                    keys.append((ds.attrs.get('index', None), name))
                hd.visit(_get_keys)
                for _, key in sorted(keys):
                    ds = hd[key]
                    parameter = self._nnp.parameter.add()
                    parameter.variable_name = key
                    parameter.shape.dim.extend(ds.shape)
                    parameter.data.extend(ds[...].flatten())
                    if ds.attrs['need_grad']:
                        parameter.need_grad = True
                    else:
                        parameter.need_grad = False

        elif e == '.protobuf':
            with open(filename, 'rb') as f:
                self._nnp.MergeFromString(f.read())

    def find_network(self, executor_name):
        net = None
        for network in self._nnp.network:
            if network.name == executor_name:
                net = network
        return net

    def find_parameter_variable(self, network):
        var_list = []
        for var in network.variable:
            if var.type == "Parameter" and var.initializer:
                var_list.append(var)
        return var_list

    def generate_parameters_data(self, var_list, batch_size):
        from nnabla.core.graph_def import _create_initializer
        from nnabla.parameter import get_parameter_or_create
        import numpy as np

        rng = np.random.RandomState(0)
        for var in var_list:
            shape = tuple(
                [d if d >= 1 else batch_size for d in var.shape.dim])
            initializer = _create_initializer(var, rng)
            variable_instance = get_parameter_or_create(
                var.name, shape, initializer)
            p = self._nnp.parameter.add()
            p.variable_name = var.name
            p.shape.dim.extend(shape)
            p.data.extend(variable_instance.d.flatten())

    def execute(self):
        self._nnp = nnabla_pb2.NNablaProtoBuf()
        other_files = []
        for ifile in self._args:
            print('Importing {}'.format(ifile))
            ext = os.path.splitext(ifile)[1].lower()
            if ext == '.nnp':
                try:
                    tmpdir = tempfile.mkdtemp()
                    with zipfile.ZipFile(ifile, 'r') as nnpzip:
                        for name in nnpzip.namelist():
                            if os.path.splitext(name)[1].lower() in ['.nntxt', '.prototxt']:
                                nnpzip.extract(name, tmpdir)
                                with open(os.path.join(tmpdir, name), 'rt') as f:
                                    text_format.Merge(f.read(), self._nnp)
                        for name in nnpzip.namelist():  # Param
                            if os.path.splitext(name)[1].lower() in ['.protobuf', '.h5']:
                                nnpzip.extract(name, tmpdir)
                                self.load_parameters(
                                    os.path.join(tmpdir, name))
                finally:
                    shutil.rmtree(tmpdir)
            elif ext in ['.nntxt', '.prototxt']:
                with open(ifile, 'rt') as f:
                    text_format.Merge(f.read(), self._nnp)
            elif ext in ['.protobuf', '.h5']:
                self.load_parameters(ifile)
            else:
                other_files.append(ifile)

        executor_name = self._nnp.executor[0].network_name
        network = self.find_network(executor_name)
        parameter_variable_list = self.find_parameter_variable(network)
        if parameter_variable_list and not self._nnp.parameter:
            self.generate_parameters_data(
                parameter_variable_list, network.batch_size)

        if self._executor_index is not None:
            if self._executor_index < len(self._nnp.executor):
                self._nnp = self._shrink_with_executor(
                    self._nnp.executor[self._executor_index])

        if self._expand_network:
            self._nnp = expander.NnpExpander(self._nnp).execute()

        class nnp:
            pass
        nnp.protobuf = self._nnp
        nnp.other_files = other_files
        return nnp
