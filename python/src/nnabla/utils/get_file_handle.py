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

import contextlib
import zipfile
import h5py
import io


@contextlib.contextmanager
def get_file_handle_load(path, ext):
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

    yield f
    if need_close:
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

    yield f
    if need_close:
        f.close()


def get_buf_type(filename):
    return filename.split('_')[-1].split('.')[1].lower()
