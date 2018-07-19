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


class OutputPath(object):
    def __init__(self, path='./tmp.output/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.path = path

    def get_filepath(self, name):
        return os.path.join(self.path, name)


_default_output_path = None


def default_output_path():
    global _default_output_path
    if _default_output_path is None:
        _default_output_path = OutputPath()
    return _default_output_path
