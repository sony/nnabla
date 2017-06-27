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

from generator_common.common import function_arguments


def generate(info, template):
    function_args = []
    function_arg_definitions = []

    for i, (category, functions) in enumerate(info['Functions'].items()):
        function_args.append('')
        function_args.append('    // {} functions'.format(category))
        for j, (function, function_info) in enumerate(functions.items()):
            arg_info = function_arguments(function_info)
            if len(arg_info['names']) > 0:
                function_args.append('    {}Parameter {}_param = {};'.format(function,
                                                                             info['Names'][
                                                                                 function],
                                                                             (i + 1) * 1000 + j + 1))
                function_arg_definitions.append('')
                function_arg_definitions.append(
                    'message {}Parameter {{'.format(function))
                for k, (t, n) in enumerate(zip(arg_info['types'], arg_info['names'])):
                    function_arg_definitions.append(
                        '  {} {} = {};'.format(t, n, k + 1))
                function_arg_definitions.append('}')

    return template.format(function_args='\n'.join(function_args),
                           function_arg_definitions='\n'.join(function_arg_definitions))
