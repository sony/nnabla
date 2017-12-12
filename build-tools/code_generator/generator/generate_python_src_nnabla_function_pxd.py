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

from generator_common.common import each_function, function_arguments
import utils.type_conv


def generate(info, template):
    definitions = []
    for function, function_info in each_function(info):
        arg_info = function_arguments(function_info)
        function_args = ''
        for arg_name, arg_type in zip(arg_info['names'], arg_info['types']):
            function_args += ', {} {}'.format(utils.type_conv.type_from_proto[
                                              arg_type]['pyx'], arg_name)
        definitions.append(
            'cdef extern from "nbla/function/{}.hpp" namespace "nbla":'.format(info['Names'][function]))
        definitions.append(
            '    shared_ptr[CFunction] create_{}(const CContext&{}) except +'.format(function, function_args))

    return template.format(function_definitions='\n'.join(definitions))
