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

import generator_common.common as common
import utils.type_conv


def generate(info, func_name, func_name_snakecase, template):

    arg_info = common.function_arguments(info)
    func_arg_variable_defines = '\n'.join(['protected:'] + ['{} {}_;'.format(utils.type_conv.type_from_proto[
                                          t]['cpp_var'], n) for t, n in zip(arg_info['types'], arg_info['names'])])
    func_arg_variable_types = ', '.join(
        [func_name] + [utils.type_conv.type_from_proto[t]['cpp'] for t in arg_info['types']])
    func_arg_initializers = ', ' + \
        ', '.join(['{0}_({0})'.format(n) for n in arg_info['names']])
    func_arg_variables = ', '.join(
        ['ctx_'] + ['{}_'.format(n) for n in arg_info['names']])
    func_args = ', '.join(['const Context &ctx'] + ['{} {}'.format(utils.type_conv.type_from_proto[
                          t]['cpp'], n) for t, n in zip(arg_info['types'], arg_info['names'])])

    io_info = common.function_io(info)
    template_defines = ', '.join(['typename {}'.format(t)
                                  for t in io_info['templates']])
    in_types = ', '.join(['get_dtype<{}>()'.format(t)
                          for t in io_info['input']['types']])
    out_types = ', '.join(['get_dtype<{}>()'.format(t)
                           for t in io_info['output']['types']])
    min_inputs = io_info['input']['min']
    min_outputs = io_info['output']['min']

    base_function_types = ', '.join(
        [utils.type_conv.type_from_proto[t]['cpp'] for t in arg_info['types']])
    base_function_args = ', '.join(
        ['ctx'] + ['{}'.format(n) for n in arg_info['names']])

    return template.format(func_name=func_name,
                           func_name_upcase=func_name.upper(),
                           template_defines=template_defines,
                           func_args=func_args,
                           func_arg_variable_defines=func_arg_variable_defines,
                           func_arg_variable_types=func_arg_variable_types,
                           func_arg_variables=func_arg_variables,
                           func_arg_initializers=func_arg_initializers,
                           in_types=in_types,
                           out_types=out_types,
                           min_inputs=min_inputs,
                           min_outputs=min_outputs,
                           base_function_types=base_function_types,
                           base_function_args=base_function_args)
