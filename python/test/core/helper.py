# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
import io
import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.testing import assert_allclose
from contextlib import contextmanager
from shutil import rmtree
import tempfile
import numpy
import warnings
from legacy_expander import NnpExpander
from nnabla.utils import nnabla_pb2
from legacy_load import _create_variable
import google.protobuf.text_format as text_format

from nnabla.utils.image_utils import imsave
from nnabla.utils.get_file_handle import get_file_handle_save
from nnabla.utils.nnp_format import nnp_version

NNP_FILE = 'tmp.nnp'


class ModuleCreator:
    def __init__(self, module, input_shape):
        self.module = module
        self.input_shape = input_shape
        self.variable_values = None
        self.proto_variable_inputs = None

    def get_variable_inputs(self):
        variable_inputs = [nn.Variable(shape) for shape in self.input_shape]
        if self.variable_values is None:
            variable_values = [np.random.random(
                shape) for shape in self.input_shape]
        for v, d in zip(variable_inputs, variable_values):
            v.d = d
        return variable_inputs

    def get_proto_variable_inputs(self):
        if self.proto_variable_inputs is None:
            proto_variable_inputs = [nn.ProtoVariable(
                shape) for shape in self.input_shape]
            self.proto_variable_inputs = proto_variable_inputs
            return proto_variable_inputs
        return self.proto_variable_inputs


def forward_variable_and_check_equal(variable_a, variable_b):
    def forward_output(variable):
        if isinstance(variable, nn.Variable):
            variable.forward()
        else:
            y = F.sink(*variable)
            for v in variable:
                v.persistent = True
            y.forward()

    for v in [variable_a, variable_b]:
        forward_output(v)
    if isinstance(variable_a, nn.Variable):
        assert_allclose(variable_a.d, variable_b.d)
        return
    for a, b in zip(variable_a, variable_b):
        assert_allclose(a.d, b.d, rtol=1e-4, atol=1e-6)


def forward_variable(inputs, outputs, side):
    rng = np.random.RandomState(389)
    if isinstance(inputs, nn.Variable):
        inputs.d = rng.randn(*inputs.d.shape)
    else:
        for v in inputs:
            v.d = rng.randn(*v.d.shape)

    if isinstance(outputs, nn.Variable):
        outputs.forward()
        yield outputs.d
    else:
        y = F.sink(*outputs)
        for v in outputs:
            v.persistent = True
        y.forward()
        for v in outputs:
            yield v.d


def iterate_function(outputs, side):
    funcs = []

    def visitor(f):
        if f.name != 'Sink':
            funcs.append(f)
    if isinstance(outputs, nn.Variable):
        outputs.visit(visitor)
    else:
        y = F.sink(*outputs)
        y.visit(visitor)

    for f in funcs:
        yield f


def assert_topology(ref_outputs, outputs):
    for ref_f, f in zip(iterate_function(ref_outputs, 'left'),
                        iterate_function(outputs, 'right')):
        assert ref_f.name == f.name
        assert ref_f.arguments == f.arguments
        for ref_v, v in zip(ref_f.inputs, f.inputs):
            assert ref_v.d.shape == v.d.shape
            assert ref_v.need_grad == v.need_grad


def assert_tensor_equal(tensor_a, tensor_ref):
    assert tensor_a.shape == tensor_ref.shape
    assert_allclose(tensor_a, tensor_ref)


@contextmanager
def create_temp_with_dir(filename):
    tmpdir = tempfile.mkdtemp()
    print('created {}'.format(tmpdir))
    csvfilename = os.path.join(tmpdir, filename)
    yield csvfilename
    rmtree(tmpdir, ignore_errors=True)
    print('deleted {}'.format(tmpdir))


@contextmanager
def generate_csv_csv(filename, num_of_data, data_size):
    with create_temp_with_dir(filename) as csvfilename:
        datadir = os.path.dirname(csvfilename)
        with open(csvfilename, 'w') as f:
            f.write('x:data,y\n')
            for n in range(0, num_of_data):
                x = numpy.ones(data_size).astype(numpy.uint8) * n
                data_name = 'data_{}.csv'.format(n)
                with open(os.path.join(datadir, data_name), 'w') as df:
                    for d in x:
                        df.write('{}\n'.format(d))
                f.write('{}, {}\n'.format(data_name, n % 10))
        yield csvfilename


@contextmanager
def generate_csv_png(num_of_data, img_size):
    with create_temp_with_dir('test.csv') as csvfilename:
        imgdir = os.path.dirname(csvfilename)
        with open(csvfilename, 'w') as f:
            f.write('x:image,y\n')
            for n in range(0, num_of_data):
                x = np.identity(img_size).astype(numpy.uint8) * n
                img_name = 'image_{}.png'.format(n)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(os.path.join(imgdir, img_name), x)
                f.write('{}, {}\n'.format(img_name, n % 10))
        yield csvfilename


@contextmanager
def create_temp_with_dir(filename):
    tmpdir = tempfile.mkdtemp()
    print('created {}'.format(tmpdir))
    csvfilename = os.path.join(tmpdir, filename)
    yield csvfilename
    rmtree(tmpdir, ignore_errors=True)
    print('deleted {}'.format(tmpdir))


def proto_from_str(nntxt_str):
    proto = nnabla_pb2.NNablaProtoBuf()
    text_format.Merge(nntxt_str, proto)
    return proto


def prepare_parameters(nntxt_str):
    proto = proto_from_str(nntxt_str)
    proto = NnpExpander(proto).execute()
    rng = np.random.RandomState(0)
    for n in proto.network:
        for v in n.variable:
            shape = tuple(
                [d if d >= 1 else n.batch_size for d in v.shape.dim])
            variable = _create_variable(v, v.name, shape, rng)


def get_input_size(proto):
    # TODO:
    return 28


@contextmanager
def generate_case_from_nntxt_str(nntxt_str, param_format, dataset_sample_num, batch_size=None):
    proto = proto_from_str(nntxt_str)
    with generate_csv_png(dataset_sample_num, get_input_size(proto)) as dataset_csv_file:
        # To test dataset, we create a randomly generated dataset.
        for ds in proto.dataset:
            ds.batch_size = batch_size if batch_size else ds.batch_size
            ds.uri = dataset_csv_file
            ds.cache_dir = os.path.join(
                os.path.dirname(dataset_csv_file), "data.cache")
        nntxt_io = io.StringIO()
        text_format.PrintMessage(proto, nntxt_io)
        nntxt_io.seek(0)

        version = io.StringIO()
        version.write('{}\n'.format(nnp_version()))
        version.seek(0)

        param = io.BytesIO()
        prepare_parameters(nntxt_str)
        nn.parameter.save_parameters(param, extension=param_format)

        with create_temp_with_dir(NNP_FILE) as temp_nnp_file_name:
            with get_file_handle_save(temp_nnp_file_name, ".nnp") as nnp:
                nnp.writestr('nnp_version.txt', version.read())
                nnp.writestr('network.nntxt', nntxt_io.read())
                nnp.writestr('parameter{}'.format(param_format), param.read())
            yield temp_nnp_file_name


@contextmanager
def get_saved_test_model(module):
    module_func, module_input_shapes = module
    with create_temp_with_dir(NNP_FILE) as nnp_file:
        with nn.graph_def.graph() as g:
            variables = [
                nn.ProtoVariable(shape) for _, shape in module_input_shapes
            ]
            outputs = module_func(*variables)
        g.save(nnp_file)
        nn.clear_parameters()
        yield nnp_file


def dump_network_topology(network):
    class Verifier:
        def __call__(self, pf):
            inputs = ','.join(pf.inputs)
            outputs = ','.join(pf.outputs)
            print("{}:{}, i:{}, o:{}".format(
                pf.type, pf.name, inputs, outputs))
    network.execute_on_proto(Verifier())
