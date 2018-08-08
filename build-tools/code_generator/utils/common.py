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

import difflib
import io
import os
import re
import six
import subprocess


def check_update(filename, generated, force=False):
    original = ''
    if os.path.exists(filename):
        with io.open(filename, 'rt', encoding='utf_8_sig') as f:
            original = six.text_type(f.read())
    s = difflib.SequenceMatcher(None, original, generated)

    if(force or not os.path.exists(filename)) and s.ratio() < 1.0:
        with open(filename, 'wb') as f:
            print('Updating {}.'.format(filename))
            write_content = generated.encode('utf_8')
            write_content = write_content.replace(b'\r\n', b'\n')
            write_content = write_content.replace(b'\r', b'\n')
            f.write(write_content)


def get_version(dir):
    os.chdir(dir)
    if "NNABLA_VERSION" in os.environ and os.environ["NNABLA_VERSION"] != "":
        version = os.environ["NNABLA_VERSION"]
        if "NNABLA_VERSION_SUFFIX" in os.environ:
            version += os.environ["NNABLA_VERSION_SUFFIX"]
        if "NNABLA_SHORT_VERSION" in os.environ:
            return version, os.environ["NNABLA_SHORT_VERSION"]
    version = default_version = '1.0.3'
    nearest_tag = default_version
    if "NNABLA_VERSION_SUFFIX" in os.environ:
            version += os.environ["NNABLA_VERSION_SUFFIX"]
    return version.replace('/', '_').lower(), nearest_tag.replace('/', '_').lower()
