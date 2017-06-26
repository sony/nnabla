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


import collections


def each_function(info):
    for category, functions in info['Functions'].items():
        for function, function_info in functions.items():
            yield((function, function_info))


def function_arguments(function_info):
    args = {'names': [], 'types': [], 'defaults': []}
    if 'argument' in function_info and len(function_info['argument']) > 0:
        for arg, arg_info in function_info['argument'].items():
            args['names'].append(arg)
            args['types'].append(arg_info['Type'])
            if 'Default' in arg_info:
                args['defaults'].append(arg_info['Default'])
            else:
                args['defaults'].append(None)
    return args


def function_io(function_info):
    ios = {'templates': [], 'template_types': [], 'template_defines': []}
    for io in ['input', 'output']:
        io_mins = 0
        io_names = []
        io_types = []
        io_ios = []
        io_params = []
        repeated_found = False
        optional_found = False
        for variable, variable_info in function_info[io].items():
            ctype = 'float'
            if repeated_found:
                raise ValueError("Field error in {}: Only last element in {} can have `repeated' field.".format(
                    function_info['name'], io))
            io_names.append(variable)
            option_list = []
            if 'Options' in variable_info:
                option_list = [x.strip()
                               for x in variable_info['Options'].split()]

            if 'Parameter' in option_list:
                io_params.append(variable)
            else:
                io_ios.append(variable)
            if 'Optional' in option_list:
                optional_found = True
            else:
                io_mins += 1

            if 'Integer' in option_list:
                ctype = 'int'

            if ctype not in ios['template_types']:
                i = len(ios['template_types'])
                template = 'T' if i == 0 else 'T{}'.format(i)
                define = template if ctype == float else '{} = {}'.format(
                    template, ctype)
                ios['templates'].append(template)
                ios['template_types'].append(ctype)
                ios['template_defines'].append(define)
            i = ios['template_types'].index(ctype)
            template = 'T' if i == 0 else 'T{}'.format(i)
            io_types.append(template)

            # Variadic
            if 'Variadic' in option_list:
                repeated_found = True

        info = collections.OrderedDict()
        info['min'] = io_mins
        info['names'] = io_names
        info['ios'] = io_ios
        info['params'] = io_params
        info['types'] = io_types
        info['repeat'] = repeated_found
        assert not optional_found or not repeated_found, "Error in {} of {}: `optional and `repeated` cannot exist together.".format(
            io, function_info['name'])
        ios[io] = info
    return ios
