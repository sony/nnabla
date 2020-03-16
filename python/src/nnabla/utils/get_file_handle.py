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


@contextlib.contextmanager
def get_file_handle_load(path, ext):
    if isinstance(path, str):
        if ext in ['.nntxt', '.prototxt']:
            need_close = True
            f = open(path, 'r')
        elif ext == '.protobuf':
            need_close = True
            f = open(path, 'rb')
        elif ext == '.nnp':
            need_close = True
            f = zipfile.ZipFile(path, 'r')
        elif ext == '.h5':
            need_close = True
            f = h5py.File(path, 'r')
        else:
            raise ValueError("Currently, ext == {} is not support".format(ext))
    else:
        if hasattr(path, 'read'):
            need_close = False
            f = path
    yield f
    if need_close:
        f.close()


@contextlib.contextmanager
def get_file_handle_save(path, ext):
    if isinstance(path, str):
        if ext in ['.nntxt', '.prototxt']:
            need_close = True
            f = open(path, 'w')
        elif ext == '.protobuf':
            need_close = True
            f = open(path, 'wb')
        elif ext == '.nnp':
            need_close = True
            f = zipfile.ZipFile(path, 'w')
        elif ext == '.h5':
            need_close = True
            f = h5py.File(path, 'w')
        else:
            raise ValueError("Currently, ext == {} is not support".format(ext))
    else:
        if hasattr(path, 'write'):
            need_close = False
            f = path
    yield f
    if need_close:
        f.close()
