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
from nnabla.utils.get_file_handle import get_file_handle_load
from nnabla.utils.get_file_handle import get_file_handle_save

# TODO temporary work around to suppress FutureWarning message.
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

current_scope = OrderedDict()
root_scope = current_scope


def get_current_parameter_scope():
    """Returns current parameter scope.
    """
    global current_scope
    return current_scope


@contextmanager
def parameter_scope(name, scope=None):
    """
    Grouping parameters registered by parametric functions
    listed in :mod:`nnabla.parametric_functions`.

    Args:

        name (str): Parameter scope name.

        scope (OrderedDict, optional):
            Specify current parameter scope as a local dictionary.
            The default value is ``None``. In this case,
            the current parameter scope maintained in global is used.

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
    if scope is None:
        scope = current_scope
    else:
        if not isinstance(scope, dict):
            raise ValueError(
                'Scope must be a dictionary. {} is given.'.format(type(scope)))
    for name in names:
        parent_scope = scope
        # When name is empty, the given scope is used as a current scope.
        if name:
            # Creates a new scope dict if it doesn't exist.
            # `dict.get` returns default value (OrderedDict())
            # if scope contains `name`
            scope = scope.get(name, OrderedDict())
            assert isinstance(scope, dict)
            parent_scope[name] = scope
    current_scope = scope
    try:
        yield current_scope
    finally:
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
    """Remove and get parameter by key.

    Args:
        key(str): Key of parameter.

    Returns: ~nnabla.Variable
        Parameter if key found, otherwise None.

    """
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


def _create_parameter_by_initializer(initializer, shape, need_grad):

    # If initializer is not set, just returns a new variable with zeros.
    if initializer is None:
        assert shape is not None
        param = nn.Variable(shape, need_grad=need_grad)
        param.data.zero()  # Initialize with zero.
        return param

    # Initialize by a numpy array.
    if isinstance(initializer, numpy.ndarray):  # numpy init
        assert (shape is None) or (tuple(shape) == initializer.shape)
        return nn.Variable.from_numpy_array(
            initializer, need_grad=need_grad)

    # Initialize by Initializer or callable object which takes shape as an argument.
    if callable(initializer):
        assert shape is not None
        return nn.Variable.from_numpy_array(
            initializer(shape=list(map(int, shape))), need_grad=need_grad)

    # Invalid initialzier argument.
    raise ValueError(
        "`initializer` must be either the :obj:`numpy.ndarray`"
        " or an instance inherited from `nnabla.initializer.BaseInitializer`.")


def get_parameter_or_create(name, shape=None, initializer=None, need_grad=True,
                            as_need_grad=None):
    """
    Returns an existing parameter variable in current parameter scope
    with the provided name.

    If a variable with the provided name does not exist,
    a new variable is created and registered to the current parameter scope
    with the name, then returned.

    Args:

        name(str):
            The name under the current scope. If it already exists, the name
            is queried from the parameter manager.
        shape (:obj:`tuple` of :obj:`int`):
            Shape of created parameter. The shape of the specified
            parameter must match with this shape. The default is None which is
            only valid if initializer is given as an :obj:`numpy.ndarray`.
        initializer (:obj:`nnabla.initializer.BaseInitializer` or :obj:`numpy.ndarray`):
            An initialization function to be applied to the parameter.
            :obj:`numpy.ndarray` can also be given to initialize parameters
            from numpy array data.
        need_grad (bool):
            Register the parameter with the specified ``need_grad`` flag.
            The default is True. If the flag is different from the previously
            specified one, the flag will be overwritten, but the values will be
            kept.
        as_need_grad (bool):
            Get a parameter variable with the specified ``need_grad`` flag.
            Note that this doesn't overwrite the flag of the registered parameter
            variable with the provided name. Instead, if the given flag
            mismatches with the previously registered ``need_grad`` flag, it
            returns a new variable referring to the same array contents but with
            ``need_grad=as_need_grad``.

    Note:
        It returns a `Variable` which is unlinked from the
        registered one in the current parmeter scope
        (using :py:meth:`nnabla.Variable.get_unlinked_variable`).
        That means changing a `need_grad` attribute doesn't affect
        the variable existing in the current parameter scope.

    """

    # Resolve delimiter '/' in parameter name.
    names = name.split('/')
    if len(names) > 1:
        with parameter_scope(names[0]):
            return get_parameter_or_create('/'.join(names[1:]), shape, initializer, need_grad, as_need_grad)

    # Set need_grad if as_need_grad is not specified.
    if as_need_grad is None:
        as_need_grad = need_grad

    # Try to find a existing parameter.
    param = get_parameter(names[0])

    # If found, verify shape and flags, and returns it.
    if param is not None:
        if param.shape != tuple(shape):
            raise ValueError(
                'The size of existing parameter "{}" {} is different from the '
                'size of new parameter {}.\n'
                'To clear all parameters, call nn.clear_parameters().'.format(
                    name, param.shape, tuple(shape)))

        if need_grad != param.need_grad:
            param.need_grad = need_grad
            set_parameter(name, param)
        return param.get_unlinked_variable(need_grad=as_need_grad)

    # TODO: Initializer info must be stored in Variable?
    # class VariableInfo:
    #     pass
    # info = VariableInfo()
    # info.initializer = initializer

    # Create a new parameter using specified configuration,
    # and write it to current scope..
    param = _create_parameter_by_initializer(initializer, shape, need_grad)
    set_parameter(name, param)
    return param.get_unlinked_variable(need_grad=as_need_grad)


def get_parameters(params=None, path='', grad_only=True):
    """Get parameter Variables under the current parameter scope.

    Args:
        params (dict): Internal use. User doesn't set it manually.
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
            parameter.variable_name, parameter.shape.dim,
            need_grad=parameter.need_grad)
        param = numpy.reshape(parameter.data, parameter.shape.dim)
        var.d = param


def load_parameters(path, proto=None, needs_proto=False, extension=".nntxt"):
    """Load parameters from a file with the specified format.

    Args:
      path : path or file object
    """
    if isinstance(path, str):
        _, ext = os.path.splitext(path)
    else:
        ext = extension

    if ext == '.h5':
        # TODO temporary work around to suppress FutureWarning message.
        import warnings
        warnings.simplefilter('ignore', category=FutureWarning)
        import h5py
        with get_file_handle_load(path, ext) as hd:
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

                var = get_parameter_or_create(
                    key, ds.shape, need_grad=ds.attrs['need_grad'])
                var.data.cast(ds.dtype)[...] = ds[...]

                if needs_proto:
                    if proto is None:
                        proto = nnabla_pb2.NNablaProtoBuf()
                    parameter = proto.parameter.add()
                    parameter.variable_name = key
                    parameter.shape.dim.extend(ds.shape)
                    parameter.data.extend(
                        numpy.array(ds[...]).flatten().tolist())
                    parameter.need_grad = False
                    if ds.attrs['need_grad']:
                        parameter.need_grad = True

    else:
        if proto is None:
            proto = nnabla_pb2.NNablaProtoBuf()

        if ext == '.protobuf':
            with get_file_handle_load(path, ext) as f:
                proto.MergeFromString(f.read())
                set_parameter_from_proto(proto)
        elif ext == '.nntxt' or ext == '.prototxt':
            with get_file_handle_load(path, ext) as f:
                text_format.Merge(f.read(), proto)
                set_parameter_from_proto(proto)

        elif ext == '.nnp':
            try:
                tmpdir = tempfile.mkdtemp()
                with get_file_handle_load(path, ext) as nnp:
                    for name in nnp.namelist():
                        nnp.extract(name, tmpdir)
                        _, ext = os.path.splitext(name)
                        if ext in ['.protobuf', '.h5']:
                            proto = load_parameters(os.path.join(
                                tmpdir, name), proto, needs_proto)
            finally:
                shutil.rmtree(tmpdir)
                logger.info("Parameter load ({}): {}".format(format, path))
        else:
            logger.error("Invalid parameter file '{}'".format(path))
    return proto


def save_parameters(path, params=None, extension=None):
    """Save all parameters into a file with the specified format.

    Currently hdf5 and protobuf formats are supported.

    Args:
      path : path or file object
      params (dict, optional): Parameters to be saved. Dictionary is of a parameter name (:obj:`str`) to :obj:`~nnabla.Variable`.
    """
    if isinstance(path, str):
        _, ext = os.path.splitext(path)
    else:
        ext = extension
    params = get_parameters(grad_only=False) if params is None else params
    if ext == '.h5':
        # TODO temporary work around to suppress FutureWarning message.
        import warnings
        warnings.simplefilter('ignore', category=FutureWarning)
        import h5py
        with get_file_handle_save(path, ext) as hd:
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

        with get_file_handle_save(path, ext) as f:
            f.write(proto.SerializeToString())
    else:
        logger.critical('Only supported hdf5 or protobuf.')
        assert False
    logger.info("Parameter save ({}): {}".format(ext, path))
