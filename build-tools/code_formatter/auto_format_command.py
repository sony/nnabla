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

#!/bin/env python

from __future__ import print_function
try:
    from builtins import str
except:
    import sys
    print("`auto_format` requires `future` package installed.", file=sys.stderr)
    raise

import io
import os
import re

import file_formatter


def _convert_file(ext, filename):
    eol = file_formatter.check_eol(filename)
    with io.open(filename, 'rt', encoding='utf_8_sig') as f:
        # 'utf_8_sig' enables to read UTF-8 formatting file with BOM
        original = str(f.read())
    converted = file_formatter.format_file(ext, original)
    write_content = str(converted)

    if not write_content == original:
        print('Formatting {}'.format(filename))
        write_content = write_content.replace('\r\n', '\n')
        write_content = write_content.replace('\r', '\n')
        write_content = write_content.replace('\n', eol)
        with open(filename, 'wt') as f:
            f.write(write_content)


def command(arg):
    # Format codes and do chmod
    for exclude in arg.exclude:
        print('Skipping files matching `{}`.'.format(exclude))
    # 0. Functions.yaml
    functions_yaml = os.path.join(arg.base, 'build-tools/code_generator/functions.yaml')
    if os.path.exists(functions_yaml):
        _convert_file('.yaml', functions_yaml)

    for root in arg.subfolder:
        for dirname, _, filenames in os.walk(os.path.join(arg.base, root)):
            for filename in filenames:
                # 1. Create full path.
                basename, extname = os.path.splitext(filename)
                extname = extname.lower()
                fullname = os.path.join(dirname, filename)
                # 2. Skip files matching exclude patterns
                excluding = False
                for exclude in arg.exclude:
                    if re.search(exclude, fullname) is not None:
                        excluding = True
                if excluding:
                    continue
                # 3. Convert Python files
                if extname in file_formatter.python_extensions:
                    if filename in file_formatter.python_exclude:
                        print('Skipped {}'.format(fullname))
                        continue
                    _convert_file(extname, fullname)
                # 4. Convert C++ files
                elif extname in file_formatter.c_extensions:
                    if filename in file_formatter.c_exclude:
                        print('Skipped {}'.format(fullname))
                        continue
                    _convert_file(extname, fullname)
                # 4. Convert C++ files
                chmod_extensions = file_formatter.c_extensions + \
                    file_formatter.python_extensions
                # 5. Set file mode
                if extname in chmod_extensions:
                    os.chmod(fullname, 0o644)

    # chmod document files
    for dirname, _, filenames in os.walk(arg.base):
        for filename in filenames:
            _, extname = os.path.splitext(filename)
            extname = extname.lower()
            fullname = os.path.join(dirname, filename)
            if os.path.islink(fullname):
                # Symlink is passed
                continue
            if extname in file_formatter.doc_extensions:
                os.chmod(fullname, 0o644)
