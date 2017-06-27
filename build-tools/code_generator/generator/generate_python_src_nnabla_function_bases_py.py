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

from generator_common.common import each_function, function_io, function_arguments
import copy

repeated_tmpl = '''
@function_api
def {function_snake}(ctx, *inputs, **args):
    r"""{description}
    """

    assert len(inputs) >= {min_inputs}, \\
        "{function_snake} must take more than {min_inputs} inputs"
    {tuple_inputs} = inputs[:{min_inputs}]
    n_outputs = args.pop('n_outputs', -1)
    outputs = args.pop('outputs', None)
{extra_block}
    return F.{function}({pass_args})(*inputs, n_outputs=n_outputs, auto_forward=auto_forward.get_auto_forward(), outputs=outputs)
'''


def generate_repeated_inputs(info, function, io_info, arg_info, description):
    input_names = io_info['input']['names']
    min_inputs = io_info['input']['min']
    repeat = io_info['input']['repeat']
    optional = min_inputs < len(input_names)
    arg_names = arg_info['names']
    arg_defaults = arg_info['defaults']
    pass_args = ['ctx'] + arg_names
    args = zip(arg_names, arg_defaults)
    args_display = []
    extra_block = []
    for n, d in args:
        if d is None:
            raise ValueError(
                "Positional args are not allowed with variable length inputs.")
        if d.__class__.__name__ == "DefaultRHS":
            drepr = d.rhs
        else:
            drepr = repr(d)
        args_display.append("{}={}".format(n, drepr))
        extra_block.append("    {} = args.pop('{}', {})".format(n, n, drepr))

    inputs_display = copy.copy(input_names)
    inputs_display[-1] = "*" + inputs_display[-1]
    return repeated_tmpl.format(
        function=function, function_snake=info['Names'][function],
        args_display=', '.join(
            inputs_display + args_display),
        extra_block='\n'.join(extra_block),
        pass_args=', '.join(pass_args),
        min_inputs=min_inputs, tuple_inputs=', '.join(input_names),
        description=description)


def generate(info, template):
    apis = []
    for function, function_info in each_function(info):
        io_info = function_io(function_info)
        arg_info = function_arguments(function_info)

        desc = function_info['description']

        variadic = False
        desc.append('Args:')
        for a_n, a_info in function_info['input'].items():
            a_type = '~nnabla.Variable'
            if 'Options' in a_info:
                if 'Variadic' in a_info['Options']:
                    variadic = True
            name = a_n
            if variadic:
                name = '*inputs'
            if 'Options' in a_info:
                desc.append('    {}({}): [{}] {}'.format(
                    name, a_type, a_info['Options'], a_info['Description']))
            else:
                desc.append('    {}({}): {}'.format(
                    name, a_type, a_info['Description']))

        if 'argument' in function_info:
            for a_n, a_info in function_info['argument'].items():

                a_type = a_info['Type']
                if a_type == 'Shape':
                    a_type = ':obj:`tuple` of :obj:`int`'
                elif a_type == 'int64':
                    a_type = 'int'
                elif a_type == 'double':
                    a_type = 'float'

                if variadic:
                    desc.append(
                        '    **param({}): [name={}] {}'.format(a_type, a_n, a_info['Description']))
                else:
                    desc.append('    {}({}): {}'.format(
                        a_n, a_type, a_info['Description']))

        desc.append('')
        desc.append('Returns:')

        for o_n, o_info in function_info['output'].items():
            desc.append('    ~nnabla.Variable: {}'.format(
                o_info['Description']))

        desc.append('')

        description = '\n    '.join(desc)

        if io_info['input']['repeat']:
            apis.append(generate_repeated_inputs(
                info, function, io_info, arg_info, description))
            continue

        args1 = []
        args2 = []
        args3 = []
        args3_optional = []
        optional_code = []
        set_default_values = []

        for i, n in enumerate(io_info['input']['names']):
            if i < io_info['input']['min']:
                args1.append(n)
                args3.append(n)
            else:
                args1.append('{} = None'.format(n))
                args3_optional.append(n)

        for n, d in zip(arg_info['names'], arg_info['defaults']):
            args2.append(n)
            if d is None:
                args1.append(n)
            else:
                if d.__class__.__name__ == "DefaultRHS":
                    args1.append('{}=None'.format(n))
                    set_default_values.append('    if {} is None:'.format(n))
                    set_default_values.append(
                        '        {} = {}'.format(n, d.rhs))
                else:
                    args1.append('{}={}'.format(n, repr(d)))

        if len(args1) > 0:
            args1 = ', ' + ', '.join(args1)
        else:
            args1 = ''
        if len(args2) > 0:
            args2 = ', ' + ', '.join(args2)
        else:
            args2 = ''
        if len(args3_optional) > 0:
            optional_code.append('    inputs = ({})'.format(', '.join(args3)))
            for a in args3_optional:
                optional_code.append('    if {} is not None:'.format(a))
                optional_code.append('        inputs += ({}, )'.format(a))
            args3 = '*inputs'
        else:
            args3 = ','.join(args3)

        apis.append('')
        apis.append('')
        apis.append('@function_api')
        apis.append('def {}(ctx{}, n_outputs=-1, outputs=None):'.format(
            info['Names'][function], args1))
        apis.append('    r"""{}'.format(description))
        apis.append('    """')
        apis.append('')
        apis += optional_code
        apis += set_default_values
        apis.append('    return F.{}(ctx{})({}, n_outputs=n_outputs, auto_forward=auto_forward.get_auto_forward(), outputs=outputs)'.format(
            function, args2, args3))
    return template.format(function_apis='\n'.join(apis))
