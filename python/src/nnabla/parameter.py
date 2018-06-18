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

from six import iteritems

from contextlib import contextmanager
from collections import OrderedDict
import google.protobuf.text_format as text_format
import numpy
import os
import shutil
import tempfile
import zipfile

import nnabla as nn
from nnabla.logger import logger
import nnabla.utils.nnabla_pb2 as nnabla_pb2

# TODO temporary work around to suppress FutureWarning message.
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

current_scope = OrderedDict()
root_scope = current_scope


@contextmanager
def parameter_scope(name):
    """
    Grouping parameters registered by parametric functions
    listed in :mod:`nnabla.parametric_functions`.

    Example:

    .. code-block:: python

        import nnabla as nn
        import nnabla.parametric_functions as PF
        import nnabla.functions as F

        with nn.parameter_scope('conv1'):
            conv_out1 = PF.convolution(x, 32, (5, 5))
            bn_out1 = PF.batch_normalization(conv_out1)
            act_out1 = F.relu(bn_out1)
        with nn.parameter_scope('conv2'):
            conv_out2 = PF.convolution(act_out1, 64, (3, 3))
            bn_out2 = PF.batch_normalization(conv_out2)
            act_out2 = F.relu(bn_out2)

    Nesting `with` blocks allows you to nest parameter scopes.
    This can also be done by using "/" inside the parameter names.

    Example:

    .. code-block:: python

        with nn.parameter_scope('network1'):
            with nn.parameter_scope('conv1'):
                conv_out1 = PF.convolution(x, 32, (5, 5))
                bn_out1 = PF.batch_normalization(conv_out1)
                act_out1 = F.relu(bn_out1)
            with nn.parameter_scope('conv2'):
                conv_out2 = PF.convolution(act_out1, 64, (3, 3))
                bn_out2 = PF.batch_normalization(conv_out2)
                act_out2 = F.relu(bn_out2)

    is equivalent to

    .. code-block:: python

        with nn.parameter_scope('network1/conv1'):
            conv_out1 = PF.convolution(x, 32, (5, 5))
            bn_out1 = PF.batch_normalization(conv_out1)
            act_out1 = F.relu(bn_out1)
        with nn.parameter_scope('network1/conv2'):
            conv_out2 = PF.convolution(act_out1, 64, (3, 3))
            bn_out2 = PF.batch_normalization(conv_out2)
            act_out2 = F.relu(bn_out2)

    """
    global current_scope
    names = name.strip('/').split('/')
    if not names:
        raise ValueError(
            'Invalid argument of parameter_scope("{}").'.format(name))
    prev_scope = current_scope
    scope = current_scope
    for name in names:
        parent_scope = scope
        # Creates a new scope dict if it doesn't exist.
        # `dict.get` returns default value (OrderedDict())
        # if scope contains `name`
        scope = scope.get(name, OrderedDict())
        assert isinstance(scope, dict)
        parent_scope[name] = scope
    current_scope = scope
    yield
    current_scope = prev_scope


def get_parameter(key):
    names = key.split('/')
    if len(names) > 1:
        with parameter_scope(names[0]):
            return get_parameter('/'.join(names[1:]))
    global current_scope
    param = current_scope.get(key, None)
    if param is not None:
        assert isinstance(param, nn.Variable)
    return param


def pop_parameter(key):
    '''Remove and get parameter by key.

    Args:
        key(str): Key of parameter.

    Returns: ~nnabla.Variable
        Parameter if key found, otherwise None.

    '''
    names = key.split('/')
    if len(names) > 1:
        with parameter_scope(names[0]):
            return pop_parameter('/'.join(names[1:]))
    global current_scope
    param = current_scope.get(key, None)
    if param is not None:
        del current_scope[key]
    return param


def set_parameter(key, param):
    names = key.split('/')
    if len(names) > 1:
        with parameter_scope(names[0]):
            return set_parameter('/'.join(names[1:]), param)
    global current_scope
    current_scope[names[0]] = param


def get_parameter_or_create(name, shape, initializer=None, need_grad=True):
    """
    Returns an existing parameter variable with the provided name.
    If a variable with the provided name does not exist,
    a new variable with the provided name is returned.

    Args:

      name(str): The name under the current scope. If it already exists, the name is queried from the
          parameter manager.
      shape (:obj:`tuple` of :obj:`int`): Shape of created parameter. The shape of the specified
          parameter must match with this shape.
      initializer (~nnabla.initializer.BaseInitializer): An initialization function to be applied to the parameter.
      need_grad (bool): The value for `need_grad` .
          The default is True.

    """
    names = name.split('/')
    if len(names) > 1:
        with parameter_scope(names[0]):
            return get_parameter_or_create('/'.join(names[1:]), shape, initializer, need_grad)
    param = get_parameter(names[0])
    if param is None:
        class VariableInfo:
            pass
        info = VariableInfo()
        info.initializer = initializer
        param = nn.Variable(shape, need_grad=need_grad)
        if initializer is not None:
            param.d = initializer(shape=param.shape)
        set_parameter(name, param)
    else:
        assert param.shape == tuple(shape)
        if need_grad != param.need_grad:
            param = param.unlinked()
            param.need_grad = need_grad
    return param


def get_parameters(params=None, path='', grad_only=True):
    """Get parameter Variables under the current parameter scope.

    Args:
        params (dict): Inernal use. User doesn't set it manually.
        path (str): Internal use.  User doesn't set it manually.
        grad_only (bool): Retrieve all parameters under the current scope if
            False, while only parameters with need_grad=True are retrieved
            if True.

    Returns:
        dict: {:obj:`str` : :obj:`~nnabla.Variable`}

    """

    global current_scope
    if params is None:
        params = OrderedDict()
    for k, v in iteritems(current_scope):
        if isinstance(v, dict):
            with parameter_scope(k):
                params = get_parameters(
                    params, '/'.join([path, k]) if path else k, grad_only=grad_only)
        else:
            assert isinstance(v, nn.Variable)
            if not grad_only or v.need_grad:
                params['/'.join([path, k]) if path else k] = v
    return params


def clear_parameters():
    """Clear all parameters in the current scope."""
    global current_scope
    for key in list(current_scope.keys()):
        del current_scope[key]


def set_parameter_from_proto(proto):
    for parameter in proto.parameter:
        var = get_parameter_or_create(
            parameter.variable_name, parameter.shape.dim)
        param = numpy.reshape(parameter.data, parameter.shape.dim)
        var.d = param
        var.need_grad = parameter.need_grad


def load_parameters(path):
    """Load parameters from a file with the specified format.

    Args:
      path : path or file object
    """
    _, ext = os.path.splitext(path)
    if ext == '.h5':
        # TODO temporary work around to suppress FutureWarning message.
        import warnings
        warnings.simplefilter('ignore', category=FutureWarning)
        import h5py
        with h5py.File(path, 'r') as hd:
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
                var = get_parameter_or_create(key, ds.shape,
                                              need_grad=ds.attrs['need_grad'])
                var.data.cast(ds.dtype)[...] = ds[...]
    elif ext == '.protobuf':
        proto = nnabla_pb2.NNablaProtoBuf()
        with open(path, 'rb') as f:
            proto.MergeFromString(f.read())
            set_parameter_from_proto(proto)
    elif ext == '.nntxt' or ext == '.prototxt':
        proto = nnabla_pb2.NNablaProtoBuf()
        with open(path, 'r') as f:
            text_format.Merge(f.read(), proto)
            set_parameter_from_proto(proto)

    elif ext == '.nnp':
        try:
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(path, 'r') as nnp:
                for name in nnp.namelist():
                    nnp.extract(name, tmpdir)
                    _, ext = os.path.splitext(name)
                    if ext in ['.protobuf', '.h5']:
                        load_parameters(os.path.join(tmpdir, name))
        finally:
            shutil.rmtree(tmpdir)
    logger.info("Parameter load ({}): {}".format(format, path))


def save_parameters(path):
    """Save all parameters into a file with the specified format.

    Currently hdf5 and protobuf formats are supported.

    Args:
      path : path or file object
    """
    _, ext = os.path.splitext(path)
    params = get_parameters(grad_only=False)
    if ext == '.h5':
        # TODO temporary work around to suppress FutureWarning message.
        import warnings
        warnings.simplefilter('ignore', category=FutureWarning)
        import h5py
        with h5py.File(path, 'w') as hd:
            params = get_parameters(grad_only=False)
            for i, (k, v) in enumerate(iteritems(params)):
                hd[k] = v.d
                hd[k].attrs['need_grad'] = v.need_grad
                # To preserve order of parameters
                hd[k].attrs['index'] = i
    elif ext == '.protobuf':
        proto = nnabla_pb2.NNablaProtoBuf()
        for variable_name, variable in params.items():
            parameter = proto.parameter.add()
            parameter.variable_name = variable_name
            parameter.shape.dim.extend(variable.shape)
            parameter.data.extend(numpy.array(variable.d).flatten().tolist())
            parameter.need_grad = variable.need_grad

        with open(path, "wb") as f:
            f.write(proto.SerializeToString())
    else:
        logger.critical('Only supported hdf5 or protobuf.')
        assert False
    logger.info("Parameter save ({}): {}".format(ext, path))
