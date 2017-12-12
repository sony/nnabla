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

from .common import each_function, function_io, function_arguments
import utils.type_conv


def generate_init_cpp(info, backend='cpu', engine='default', rank=0):
    includes = []
    registers = []
    engine_suffix = '' if engine == 'default' else engine
    function_name_suffix = '' if backend == 'cpu' else backend.capitalize() + \
        engine_suffix.capitalize()
    for function, function_info in each_function(info):
        if 'Implements' not in info or (function in info['Implements'] and
                                        ((backend in info['Implements'][function] and engine == 'default') or (engine in info['Implements'][function]))):
            includes.append('{}.hpp'.format(info['Names'][function]))
            io_info = function_io(function_info)
            arg_info = function_arguments(function_info)
            function_args = ''
            for arg_type in arg_info['types']:
                function_args += ', ' + \
                    utils.type_conv.type_from_proto[arg_type]['cpp']
            function_suffix = ''.join(
                [x[0] for x in io_info['template_types']])
            registers.append('  typedef {0}{1}<{2}> {0}{1}{3};'.format(function,
                                                                       function_name_suffix,
                                                                       ', '.join(
                                                                           io_info['template_types']),
                                                                       function_suffix))
            registers.append('  NBLA_REGISTER_FUNCTION_IMPL({0}, {0}{1}{2}, {3}, "{4}", "{5}"{6});'.format(function,
                                                                                                           function_name_suffix,
                                                                                                           function_suffix,
                                                                                                           rank,
                                                                                                           backend,
                                                                                                           engine,
                                                                                                           function_args))
    return includes, registers
