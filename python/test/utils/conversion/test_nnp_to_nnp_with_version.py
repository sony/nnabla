# Copyright 2022 Sony Group Corporation.
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
import tempfile
import warnings
import pytest
import numpy
import numpy as np
from shutil import rmtree
from contextlib import contextmanager
import nnabla as nn
import nnabla.utils.converter as cvt
from nnabla.utils import nnabla_pb2
from nnabla.utils.image_utils import imsave
import google.protobuf.text_format as text_format
from nnabla.utils.nnp_format import nnp_version
from nnabla.parameter import get_parameter_or_create
from nnabla.utils.get_file_handle import get_file_handle_save
from nnabla.utils.converter.nnabla import NnpExpander
from nnabla.initializer import (
    NormalInitializer, UniformInitializer, ConstantInitializer, RangeInitializer,
    calc_normal_std_he_forward, calc_normal_std_he_backward, calc_normal_std_glorot, calc_uniform_lim_glorot)
from .nntxt import (N0001, N0002, N0003, N0004, N0005)


N_ARRAY = [N0001, N0002, N0003, N0004, N0005]


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


def get_input_size(proto):
    # TODO:
    return 28


def _create_variable(v, name, shape, rng):
    # Create and initialize variables
    class Variable:
        pass

    parameter = v.type == "Parameter"
    variable_instance = None
    if parameter:
        if v.initializer.type == 'Normal':
            initializer = NormalInitializer(v.initializer.multiplier, rng=rng)
        elif v.initializer.type == 'NormalAffineHe' or v.initializer.type == 'NormalAffineHeForward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_forward(
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalAffineHeBackward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_backward(
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalAffineGlorot':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_glorot(
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalConvolutionHe' or v.initializer.type == 'NormalConvolutionHeForward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_forward(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalConvolutionHeBackward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_backward(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalConvolutionGlorot':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_glorot(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalCLConvHe' or v.initializer.type == 'NormalCLConvHeForward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_forward(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalCLConvHeBackward':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_he_backward(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'NormalCLConvGlorot':
            initializer = (lambda shape: NormalInitializer(calc_normal_std_glorot(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'Uniform':
            initializer = UniformInitializer(
                lim=[-v.initializer.multiplier, v.initializer.multiplier], rng=rng)
        elif v.initializer.type == 'UniformAffineGlorot':
            initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
                shape[0], numpy.prod(shape[1:])), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'UniformConvolutionGlorot':
            initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
                shape[-3], shape[0], kernel=shape[-2:]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'UniformCLConvGlorot':
            initializer = (lambda shape: UniformInitializer(calc_uniform_lim_glorot(
                shape[-1], shape[0], kernel=shape[1:3]), rng=rng)(shape) * v.initializer.multiplier)
        elif v.initializer.type == 'Range':
            initializer = (lambda shape: RangeInitializer(0, 1)
                           (shape) * v.initializer.multiplier)
        elif v.initializer.type == 'Constant':
            initializer = ConstantInitializer(value=v.initializer.multiplier)
        else:
            initializer = None
        variable_instance = get_parameter_or_create(name, shape, initializer)
    else:
        # create empty variable, memory will be allocated in network.setup()
        # after network optimization
        variable_instance = nn.Variable()

    variable = Variable()
    variable.name = name
    variable.parameter = parameter
    variable.shape = shape
    variable.variable_instance = variable_instance

    return variable


def prepare_parameters(nntxt_str):
    proto = proto_from_str(nntxt_str)
    proto = NnpExpander(proto).execute()
    rng = np.random.RandomState(0)
    for n in proto.network:
        for v in n.variable:
            shape = tuple(
                [d if d >= 1 else n.batch_size for d in v.shape.dim])
            variable = _create_variable(v, v.name, shape, rng)


def nnp_file_name():
    return f"{nn.__version__}.nnp"


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

        with create_temp_with_dir(nnp_file_name()) as temp_nnp_file_name:
            with get_file_handle_save(temp_nnp_file_name, ".nnp") as nnp:
                nnp.writestr('nnp_version.txt', version.read())
                nnp.writestr('network.nntxt', nntxt_io.read())
                nnp.writestr('parameter{}'.format(param_format), param.read())
            yield temp_nnp_file_name


def set_default_value(args):
    args.api = -1
    args.batch_size = 1
    args.channel_last = False
    args.config = None
    args.dataset = None
    args.default_variable_type = ['FLOAT32']
    args.define_version = None
    args.enable_optimize_pd = False
    args.export_format = 'NNP'
    args.force = False
    args.import_format = 'NNP'
    args.inputs = None
    args.mpi = False
    args.nnp_exclude_parameter = False
    args.nnp_exclude_preprocess = False
    args.nnp_import_executor_index = None
    args.nnp_no_expand_network = False
    args.nnp_parameter_h5 = False
    args.nnp_parameter_nntxt = False
    args.outputs = None
    args.quantization = False
    args.settings = None
    args.split = None


@pytest.mark.parametrize("nn_version", ["1.12.0"])
@pytest.mark.parametrize("nntxt_str", [N0001, N0003])
def test_nnp_to_nnp_with_version_unsupported(nn_version, nntxt_str):
    class Args:
        pass
    args = Args()
    set_default_value(args)
    with pytest.raises(ValueError) as e:
        with generate_case_from_nntxt_str(nntxt_str, ".h5", 32) as nnp_file:
            dirname = os.path.dirname(nnp_file)
            out_file = os.path.join(dirname, f"{nn_version}.nnp")
            args.files = [nnp_file]
            args.nnp_version = nn_version
            ifiles = [nnp_file]
            output = out_file

            cvt.convert_files(args, ifiles, output)
        print(e)


@pytest.mark.parametrize("nn_version", ["1.12.0"])
@pytest.mark.parametrize("nntxt_idx", [1, 3, 4])
def test_nnp_to_nnp_with_version_supported(nn_version, nntxt_idx):
    class Args:
        pass
    args = Args()
    set_default_value(args)
    nntxt_str = N_ARRAY[nntxt_idx]
    with generate_case_from_nntxt_str(nntxt_str, ".h5", 32) as nnp_file:
        dirname = os.path.dirname(nnp_file)
        out_file = os.path.join(dirname, f"{nn_version}.nnp")
        args.files = [nnp_file]
        args.nnp_version = nn_version
        ifiles = [nnp_file]
        output = out_file

        assert cvt.convert_files(args, ifiles, output), "conversion failed!"
