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

type_from_proto = {
    'Shape': {'cpp': 'const vector<int> &', 'cpp_var': 'const vector<int>', 'pyx': 'const vector[int]&', 'pxd': 'const vector[int]&'},
    'int64': {'cpp': 'int', 'cpp_var': 'int', 'pyx': 'int', 'pxd': 'int'},
    'bool': {'cpp': 'bool', 'cpp_var': 'bool', 'pyx': 'cpp_bool', 'pxd': 'cpp_bool'},
    'float': {'cpp': 'float', 'cpp_var': 'float', 'pyx': 'float', 'pxd': 'float'},
    'double': {'cpp': 'double', 'cpp_var': 'double', 'pyx': 'double', 'pxd': 'double'},
    'repeated int64': {'cpp': 'const vector<int> &', 'cpp_var': 'const vector<int>', 'pyx': 'const vector[int]&', 'pxd': 'const vector[int]&'},
    'repeated float': {'cpp': 'const vector<float> &', 'cpp_var': 'const vector<float>', 'pyx': 'const vector[float]&', 'pxd': 'const vector[float]&'},
    'string': {'cpp': 'const string &', 'cpp_var': 'const string', 'pyx': 'const string&', 'pxd': 'const string&'},
    'Communicator': {'cpp': 'const shared_ptr<Communicator> &', 'cpp_var': 'shared_ptr<const Communicator>', 'pyx': 'Communicator', 'pxd': 'shared_ptr[CCommunicator]&'}
}
