# Copyright 2020,2021 Sony Corporation.
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

import contextlib
import io
import os
import re
import zipfile
from collections import OrderedDict
from functools import partial

import google.protobuf.text_format as text_format
import h5py
import nnabla as nn
import numpy
from nnabla.logger import logger
from nnabla.utils import nnabla_pb2
from nnabla.utils.nnp_format import nnp_version


class FileHandlerContext:
    pass


@contextlib.contextmanager
def get_file_handle_load(nnp, path, ext):
    if nnp is None:
        if ext == '.nnp':
            need_close = True
            f = zipfile.ZipFile(path, 'r')
        elif ext == '.h5':
            need_close = True
            if isinstance(path, str):
                f = h5py.File(path, 'r')
            else:
                f = h5py.File(io.BytesIO(path.read()), 'r')
        elif ext in ['.nntxt', '.prototxt']:
            if hasattr(path, 'read'):
                need_close = False
                f = path
            else:
                need_close = True
                f = open(path, 'r')
        elif ext == '.protobuf':
            if hasattr(path, 'read'):
                need_close = False
                f = path
            else:
                need_close = True
                f = open(path, 'rb')
        else:
            raise ValueError("Currently, ext == {} is not support".format(ext))

        try:
            yield f
        finally:
            if need_close:
                f.close()
    else:
        if ext in ['.h5']:
            # if nnp is not None and extension is .h5
            # we assume to return a .h5 type file handler
            with nnp.open(path, 'r') as n:
                f = h5py.File(io.BytesIO(n.read()), 'r')
                yield f
        else:
            f = nnp.open(path, 'r')
            try:
                yield f
            finally:
                f.close()


@contextlib.contextmanager
def get_file_handle_save(path, ext):
    if ext == '.nnp':
        need_close = True
        f = zipfile.ZipFile(path, 'w')
    elif ext == '.h5':
        need_close = True
        f = h5py.File(path, 'w')
    elif ext in ['.nntxt', '.prototxt']:
        if hasattr(path, 'read'):
            need_close = False
            f = path
        else:
            need_close = True
            f = open(path, 'w')
    elif ext == '.protobuf':
        if hasattr(path, 'read'):
            need_close = False
            f = path
        else:
            need_close = True
            f = open(path, 'wb')
    else:
        raise ValueError("Currently, ext == {} is not support".format(ext))
    try:
        yield f
    finally:
        if need_close:
            f.close()
        if hasattr(path, 'seek'):
            path.seek(0)


def get_buf_type(filename):
    return filename.split('_')[-1].split('.')[1].lower()


# TODO: Here is some known problems.
#   - Even when protobuf file includes network structure,
#     it will not loaded.
#   - Even when prototxt file includes parameter,
#     it will not loaded.
def _nntxt_file_loader(ctx, file_loaders, nnp, filename, ext):
    '''.nntxt, .prototxt
    nntxt file loader
    This loader only loads .nntxt or .prototxt files
    '''
    if not ctx.parameter_only:
        with get_file_handle_load(nnp, filename, ext) as f:
            try:
                text_format.Merge(f.read(), ctx.proto)
            except:
                logger.critical('Failed to read {}.'.format(filename))
                logger.critical(
                    '2 byte characters may be used for file name or folder name.')
                raise
    if len(ctx.proto.parameter) > 0:
        if not ctx.exclude_parameter:
            _nntxt_parameter_file_loader(ctx, file_loaders, nnp, filename, ext)


def _parameter_file_loader(ctx, file_loaders, nnp, filename, ext):
    '''.protobuf, .h5
    protobuf file loader
    This handler only handles .protobuf or .h5 files
    Currently, .h5 files are always treated as parameter file
    '''
    if not ctx.exclude_parameter:
        if ext == ".h5":
            _h5_parameter_file_loader(ctx, file_loaders, nnp, filename, ext)
        elif ext == ".protobuf":
            _pb_parameter_file_loader(ctx, file_loaders, nnp, filename, ext)
        else:
            raise ValueError("Not support extension: {}".format(ext))


def _nnp_file_loader(ctx, file_loaders, nnp, filename, ext):
    '''.nnp
    nnp file loader
    This loader only handles .nnp file, which
    call other handlers to help to digest files.
    '''
    assert nnp == None, ".nnp in a .nnp file seems impossible."
    with get_file_handle_load(nnp, filename, ext) as n:
        for name in n.namelist():
            _, ext = os.path.splitext(name)
            # version.txt is omitted since no handler for it.
            # if name == 'nnp_version.txt':
            #    pass # TODO: Currently do nothing with version
            for supported_extensions, file_loader in file_loaders.items():
                if ext in supported_extensions:
                    file_loader(ctx, file_loaders, n, name, ext)


def load_solve_state_from_h5(nnp, filename):
    '''Load solve state from h5 file
    This code is duplicated with solver.pyx, since solver.pyx only accept a string file
    as its filename input, not support ByteIO style input. But supporting ByteIO is
    important for high performance requirement.
    '''
    class SolverState:
        def __init__(self):
            self.t = 0
            self.pstate = {}

    states = OrderedDict()
    with get_file_handle_load(nnp, filename, '.h5') as f:
        # File's key is `/{parameter-name}/{state-name}` and `/{parameter-name}/t`
        skeys = []
        pkeys = set()

        def _get_skeys(name, obj):
            if not isinstance(obj, h5py.Dataset):
                # Group
                return
            # state key
            skeys.append(name)
            # To preserve order of parameters
            index = obj.parent.attrs.get('index', None)
            pname = name[:name.rindex("/")]
            # parameter key
            pkeys.add((index, pname))

        # TODO Refactoring visit logic for efficient execution
        f.visititems(_get_skeys)
        for _, pkey in sorted(pkeys):
            state = SolverState()
            for skey in skeys:
                if not skey.startswith(pkey):
                    continue
                ds = f[skey]
                if skey.endswith("/t"):
                    state.t = ds[...]
                else:
                    sname = skey.split("/")[-1]
                    var = nn.Variable.from_numpy_array(ds[...])
                    state.pstate[sname] = var
            states[pkey] = state
    return states


def _opti_file_loader(ctx, fileloaders, nnp, filename, ext):
    '''.optimizer
    optimizer file loader
    This loader only handles .optimizer file.
    '''
    file_type = get_buf_type(filename)
    if file_type == 'protobuf':
        opti_proto = nnabla_pb2.NNablaProtoBuf()
        with get_file_handle_load(nnp, filename, '.protobuf') as f:
            opti_proto.MergeFromString(f.read())
        for p_opti in opti_proto.optimizer:
            o = ctx.optimizers.get(p_opti.name, None)
            if o:
                o.solver.set_states_from_protobuf(p_opti)
            else:
                logger.warning(
                    'No matched optimizer is found for {}.'.format(filename))
    elif file_type == 'h5':
        loaded = False
        for o in ctx.optimizers.values():
            key = '{}_{}_optimizer.h5.optimizer'.format(
                o.name,
                re.sub(r'(|Cuda)$', '', str(o.solver.name))
            )
            if key == filename:
                o.solver.set_states(load_solve_state_from_h5(nnp, filename))
                loaded = True
        if not loaded:
            logger.warning(
                "No matched optimizer is found for {}.".format(filename))


def _opti_file_rough_loader(ctx, fileloaders, nnp, filename, ext):
    '''.optimizer
    optimizer solver checkpoint file rough loader
    This loader loads solver state and allow user to decide when to restore it
    '''
    file_type = get_buf_type(filename)
    optimizer_states = OrderedDict()
    if file_type == 'protobuf':
        opti_proto = nnabla_pb2.NNablaProtoBuf()
        with get_file_handle_load(nnp, filename, '.protobuf') as f:
            opti_proto.MergeFromString(f.read())
        for p_opti in opti_proto.optimizer:
            optimizer_states[p_opti.name] = ('.protobuf', p_opti)
    elif file_type == 'h5':
        h5fio = io.BytesIO()
        h5fio.name = f"{'_'.join(filename.split('_')[:2])}.h5"
        with nnp.open(filename, 'r') as f:
            h5fio.write(f.read())
            h5fio.seek(0)
        optimizer_states[filename.split('_')[0]] = ('.h5', h5fio)
    ctx.optimizer_states_checkpoint = optimizer_states


def _h5_parameter_file_loader(ctx, file_loader, nnp, filename, ext):
    with get_file_handle_load(nnp, filename, ext) as hd:
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

            var = nn.parameter.get_parameter_or_create(
                key, ds.shape, need_grad=ds.attrs['need_grad'])
            var.data.cast(ds.dtype)[...] = ds[...]

            if hasattr(ctx, "needs_proto") and ctx.needs_proto:
                parameter = ctx.proto.parameter.add()
                parameter.variable_name = key
                parameter.shape.dim.extend(ds.shape)
                parameter.data.extend(
                    numpy.array(ds[...]).flatten().tolist())
                parameter.need_grad = False
                if ds.attrs['need_grad']:
                    parameter.need_grad = True


def _pb_parameter_file_loader(ctx, file_loaders, nnp, filename, ext):
    with get_file_handle_load(nnp, filename, ext) as f:
        try:
            ctx.proto
        except:
            ctx.proto = nnabla_pb2.NNablaProtoBuf()
        ctx.proto.MergeFromString(f.read())
        nn.parameter.set_parameter_from_proto(ctx.proto)


def _nntxt_parameter_file_loader(ctx, file_loaders, nnp, filename, ext):
    with get_file_handle_load(nnp, filename, ext) as f:
        text_format.Merge(f.read(), ctx.proto)
        nn.parameter.set_parameter_from_proto(ctx.proto)


def _nnp_parameter_file_loader(ctx, file_loaders, nnp, filename, ext):
    return _nnp_file_loader(ctx, file_loaders, nnp, filename, ext)


def get_parameter_file_loader():
    file_loaders = OrderedDict([
        ('.h5', _h5_parameter_file_loader),
        ('.protobuf', _pb_parameter_file_loader),
        ('.nntxt,.prototxt', _nntxt_parameter_file_loader),
        ('.nnp', _nnp_parameter_file_loader)
    ])
    return file_loaders


def get_initial_file_loader():
    '''get file loader for first stage loading
    Some loaders are used to establish skelecton of whole nnabla graph,
    they should be performed for this first stage.
    '''
    file_loaders = OrderedDict([
        ('.nntxt,.prototxt', _nntxt_file_loader),
        ('.protobuf,.h5', _parameter_file_loader),
        ('.nnp', _nnp_file_loader),
        ('.optimizer', _opti_file_rough_loader)
    ])
    return file_loaders


def get_decorated_file_loader():
    '''get file loader for second stage loading
    Second stage means graph, optimizer, executor, ... and
    so on has already been loaded, the files to load are used
    to decorate existing nnabla representation.
    '''
    file_loaders = OrderedDict([
        ('.optimizer', _opti_file_loader),
        ('.protobuf,.h5', _parameter_file_loader),
        ('.nnp', _nnp_parameter_file_loader)
    ])
    return file_loaders


def load_files(ctx, file_loaders, filenames, extension=None):
    '''load_files
    Load files, if filesnames is not a list, we converted it to a list.
    If filenames is a list, we handle it one by one, with the handler tied with
    its postfix name.

    Args:
        ctx: A object that represents the context of sharing states with loaders.
        file_loaders (OrderedDict): List of handlers, tied with extension name.
        filenames (list): List of filenames.
        extension (str): File extension name, used to identify the file type
    Returns:
        None
    '''
    def _load_files():
        for filename in filenames:
            if isinstance(filename, str):
                _, ext = os.path.splitext(filename)
            else:
                ext = extension

            handled = False
            for supported_extensions, file_loader in file_loaders.items():
                if ext in supported_extensions:
                    file_loader(ctx, file_loaders, None, filename, ext)
                    handled = True
            else:
                if not handled:
                    logger.warning('{} is omitted.'.format(filename))

    if isinstance(filenames, list) or isinstance(filenames, tuple):
        pass
    elif isinstance(filenames, str) or hasattr(filenames, 'read'):
        filenames = [filenames]

    if hasattr(ctx, 'parameter_scope'):
        with nn.parameter_scope('', ctx.parameter_scope):
            _load_files()
    else:
        _load_files()


def _nntxt_file_saver(ctx, filename, ext):
    logger.info("Saving {} as prototxt".format(filename))
    with get_file_handle_save(filename, ext) as f:
        text_format.PrintMessage(ctx.proto, f)


def _protobuf_file_saver(ctx, filename, ext):
    logger.info("Saving {} as protobuf".format(filename))
    with get_file_handle_save(filename, ext) as f:
        f.write(ctx.proto.SerializeToString())


def _nnp_file_saver(ctx, filename, ext, ssf=None):
    logger.info("Saving {} as nnp".format(filename))
    if ssf and ssf not in ['.h5', '.protobuf']:
        logger.error(
            f"{ssf} format is not supported when save optimizer solver state.")

    nntxt = io.StringIO()
    _nntxt_file_saver(ctx, nntxt, ".nntxt")

    version = io.StringIO()
    version.write('{}\n'.format(nnp_version()))
    version.seek(0)

    param = io.BytesIO()
    if ctx.parameters is None:
        nn.parameter.save_parameters(param, extension='.h5')
    else:
        nn.parameter.save_parameters(
            param, ctx.parameters, extension='.h5')
    if ssf:
        from nnabla.utils.cli.utility import save_optimizer_states
        filenamebase = os.path.splitext(filename)[0]
        opti_filenames = save_optimizer_states(
                filenamebase, ssf, ctx)

    with get_file_handle_save(filename, ext) as nnp:
        nnp.writestr('nnp_version.txt', version.read())
        nnp.writestr('network.nntxt', nntxt.read())
        nnp.writestr('parameter.h5', param.read())
        if ssf:
            for f in opti_filenames:
                nnp.write(f, f[len(filenamebase) + 1:])

    if ssf:
        for f in opti_filenames:
            os.unlink(f)


def _h5_parameter_file_saver(ctx, filename, ext):
    with get_file_handle_save(filename, ext) as hd:
        for i, (k, v) in enumerate(ctx.parameters.items()):
            hd[k] = v.data.get_data("r")
            hd[k].attrs['need_grad'] = v.need_grad
            # To preserve order of parameters
            hd[k].attrs['index'] = i


def _protobuf_parameter_file_saver(ctx, filename, ext):
    proto = nnabla_pb2.NNablaProtoBuf()
    for variable_name, variable in ctx.parameters.items():
        parameter = proto.parameter.add()
        parameter.variable_name = variable_name
        parameter.shape.dim.extend(variable.shape)
        parameter.data.extend(numpy.array(
            variable.data.get_data("r")).flatten().tolist())
        parameter.need_grad = variable.need_grad
    with get_file_handle_save(filename, ext) as f:
        f.write(proto.SerializeToString())


def get_default_file_savers(solver_state_format=None):
    '''get_default_file_savers
    '''
    file_savers = OrderedDict([
        ('.nntxt, .prototxt', _nntxt_file_saver),
        ('.protobuf', _protobuf_file_saver),
        ('.nnp', partial(_nnp_file_saver, ssf=solver_state_format))
    ])
    return file_savers


def get_parameter_file_savers():
    '''get_parameter_file_savers
    '''
    file_savers = OrderedDict([
        ('.h5', _h5_parameter_file_saver),
        ('.protobuf', _protobuf_parameter_file_saver)
    ])
    return file_savers


def save_files(ctx, file_savers, filenames, extension=None):
    '''save_file
    Save file, if filename is a ByteIO object, extension should be set
    Args:
        ctx: A object that represents the context of sharing states with loaders.
        file_savers (OrderedDict): List of handlers, tied with extension name.
        filename (str or ByteIO or str list): filename or filename list or file object
        extension (str): File extension name, used to identify the file type,
                         only filename is file object, it does work.
    Returns:
        None
    '''
    if isinstance(filenames, list) or isinstance(filenames, tuple):
        pass
    elif isinstance(filenames, str) or hasattr(filenames, 'read'):
        filenames = [filenames]

    file_supported = []
    for filename in filenames:
        if isinstance(filename, str):
            _, ext = os.path.splitext(filename)
        else:
            ext = extension

        supported = False
        for supported_ext, file_saver in file_savers.items():
            if ext in supported_ext:
                file_saver(ctx, filename, ext)
                supported = True
                break
        file_supported.append(supported)
    return all(file_supported)
