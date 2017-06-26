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


def generate(info, template):
    create_functions = []
    if_prefix = ''
    for function, function_info in each_function(info):
        arg_info = function_arguments(function_info)
        create_functions.append(
            '    {}if f.type == "{}":'.format(if_prefix, function))
        args = ['ctx']
        for arg, type in zip(arg_info['names'], arg_info['types']):
            arg_suffix = '.dim' if type == 'Shape' else ''
            args.append('{0}=f.{1}_param.{0}{2}'.format(
                arg, info['Names'][function], arg_suffix))
        create_functions.append('        function = F.{}({})'.format(
            function, ',\n            '.join(args)))
        if_prefix = 'el'
    return template.format(create_functions='\n'.join(create_functions))
