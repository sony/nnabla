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
        function_args1 = ''
        function_args2 = ''
        store_args = ['    info.args = {}']
        for arg_name, arg_type in zip(arg_info['names'], arg_info['types']):
            function_args1 += ', {} {}'.format(utils.type_conv.type_from_proto[
                                               arg_type]['pyx'], arg_name)
            function_args2 += ', {}'.format(arg_name)
            store_args.append("    info.args['{0}'] = {0}".format(arg_name))
        definitions.append('def {}(CContext ctx{}):'.format(
            function, function_args1))
        definitions.append('    info = Info()')
        definitions += store_args
        definitions.append('    info.type_name = \'{}\''.format(function))
        definitions.append('    f = Function.create(create_{}(ctx{}), info)'.format(
            function, function_args2))
        definitions.append('    return f')
    return template.format(function_definitions='\n'.join(definitions))
