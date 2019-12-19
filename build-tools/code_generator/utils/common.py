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
import six


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
