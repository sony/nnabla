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

import os
import subprocess


c_extensions = ['.c', '.h', '.cpp', '.hpp', '.cxx', '.hxx', '.cu', '.cuh']
c_exclude = [
]

python_extensions = ['.py', '.pxd', '.pyx']
python_exclude = [
    'nnabla_pb2.py'
]

# Files with the following extensions will have 644 file permission.
doc_extensions = ['.md', '.rst', '.txt', '.toc']


def check_eol(filename):
    eol = b'\n'
    with open(filename, 'rb') as f:
        d = f.read()
        if b'\r\n' in d:
            eol = b'\r\n'
        elif b'\n' in d:
            eol = b'\n'
        elif b'\r' in d:
            eol = b'\r'
    return eol


def which(name):
    for p in os.environ['PATH'].split(os.pathsep):
        f = os.path.join(p, name)
        if os.path.isfile(f):
            return f
        if os.name is 'nt':
            p = p.replace('"', '')
            f = os.path.join(p, name)
            if os.path.isfile(f):
                return f
            f = f + '.exe'
            if os.path.isfile(f):
                return f
    return None


def search_clang_format():
    base = 'clang-format'
    # versions = ['3.4', '3.5', '3.6', '3.7', '3.8', '3.9']
    versions = ['3.8']  # Use clang-format-3.8
    for c in [base] + [base + '-{}'.format(v) for v in versions]:
        clang = which(c)
        if clang is not None:
            return clang
    raise ValueError("Not found: clang-format-3.8")


def search_autopep8():
    autopep8 = which('autopep8')
    if autopep8 is None:
        raise ValueError("Not found: autopep8")
    return autopep8


def format_file(file_ext, input):
    cmd = None
    file_ext = file_ext.lower()
    if file_ext in c_extensions:
        cmd = [search_clang_format(), '--style=llvm']
    elif file_ext in python_extensions:
        cmd = [search_autopep8(), '-']
    else:
        return input
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    output, output_err = p.communicate(input.encode('utf-8'))
    return output.decode('utf_8')
