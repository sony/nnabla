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
    include_lines = []
    creator_lines = ['// Function creator']

    for i, (category, functions) in enumerate(info['Functions'].items()):
        for j, (function, function_info) in enumerate(functions.items()):
            creator_lines.append(
                'if( func.type() == "{}") {{'.format(function))
            arg_info = function_arguments(function_info)
            include_lines.append(
                '#include <nbla/function/{}.hpp>'.format(info['Names'][function]))

            if len(arg_info['names']) > 0:
                creator_lines.append('    {}Parameter param = func.{}_param();'.format(
                    function, info['Names'][function]))
                args = []
                arg_index = 1
                for arg_name, arg_type in zip(arg_info['names'], arg_info['types']):
                    av = 'arg{}'.format(arg_index)
                    if arg_type == 'repeated int64':
                        creator_lines.append(
                            '    std::vector<int> {};'.format(av))
                        creator_lines.append(
                            '    for( int i = 0; i < param.{}_size(); i++) {{'.format(arg_name))
                        creator_lines.append(
                            '        {}.push_back(param.{}(i));'.format(av, arg_name))
                        creator_lines.append('    }')
                    elif arg_type == 'Shape':
                        creator_lines.append(
                            '    std::vector<int> {};'.format(av))
                        creator_lines.append('    {')
                        creator_lines.append(
                            '        Shape s = param.{}();'.format(arg_name))
                        creator_lines.append(
                            '        for( int i = 0; i < s.dim_size(); i++) {')
                        creator_lines.append(
                            '            {}.push_back(s.dim(i));'.format(av))
                        creator_lines.append('        }')
                        creator_lines.append('    }')
                    elif arg_type == 'double':
                        creator_lines.append(
                            '    double {} = param.{}();'.format(av, arg_name))
                    elif arg_type == 'float':
                        creator_lines.append(
                            '    float {} = param.{}();'.format(av, arg_name))
                    elif arg_type == 'int64':
                        creator_lines.append(
                            '    int {} = param.{}();'.format(av, arg_name))
                    elif arg_type == 'bool':
                        creator_lines.append(
                            '    bool {} = param.{}();'.format(av, arg_name))
                    elif arg_type == 'string':
                        creator_lines.append(
                            '    std::string {} = param.{}();'.format(av, arg_name))
                    args.append(av)
                    arg_index += 1

                creator_lines.append('    nbla::FunctionPtr fp = create_{}(ctx_, {});'.format(
                    function, ' ,'.join(args)))
            else:
                creator_lines.append(
                    '    nbla::FunctionPtr fp = create_{}(ctx_);'.format(function))
            creator_lines.append(
                '    return std::make_shared<nbla::CgFunction>(fp);')

            creator_lines.append('}')

    creator_lines.append('// End of function creator')

    function_includes = "\n".join(include_lines)
    function_creator = "\n            ".join(creator_lines)

    return template.format(function_includes=function_includes,
                           function_creator=function_creator)
