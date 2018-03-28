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

import io
import os
from os.path import abspath, dirname, join
from collections import OrderedDict

from utils.load_function_rst import Functions
from utils.common import check_update
import code_generator_utils as utils


def input_yaml(inp):
    y = OrderedDict()
    y['doc'] = inp['Description']
    if 'Options' not in inp:
        return y
    options = inp['Options'].split()
    for opt in options:
        if opt == 'Integer':
            y['template'] = 'TI'
            continue
        y[opt.lower()] = True
    return y


def inputs_yaml(inputs):
    y = OrderedDict()
    for k, v in inputs.items():
        y[k] = input_yaml(v)
    return y


def output_yaml(inp):
    y = OrderedDict()
    y['doc'] = inp['Description']
    if 'Options' not in inp:
        return y
    options = inp['Options'].split()
    for opt in options:
        if opt == 'Integer':
            y['template'] = 'TI'
            continue
        y[opt.lower()] = True
    return y


def outputs_yaml(outputs):
    y = OrderedDict()
    for k, v in outputs.items():
        y[k] = output_yaml(v)
    return y


def argument_yaml(arg):
    y = OrderedDict()
    y['doc'] = arg['Description']
    y['type'] = arg['Type']
    if 'Default' in arg:
        try:
            y['default'] = arg['Default'].rhs
        except:
            y['default'] = repr(arg['Default'])
    return y


def arguments_yaml(args):
    y = OrderedDict()
    for k, v in args.items():
        y[k] = argument_yaml(v)
    return y


def function_yaml(func, snake_name):
    func_yaml = OrderedDict()
    func_yaml['snake_name'] = snake_name
    func_yaml['doc'] = '\n'.join(func['description'])
    func_yaml['inputs'] = inputs_yaml(func['input'])
    if 'argument' in func:
        func_yaml['arguments'] = arguments_yaml(func['argument'])
    func_yaml['outputs'] = outputs_yaml(func['output'])
    return func_yaml


def category_yaml(cat, names):
    cat_yaml = OrderedDict()
    for func_name, func in cat.items():
        cat_yaml[func_name] = function_yaml(func, names[func_name])
    return cat_yaml


def main():
    functions = Functions()
    info = functions.info

    root_yaml = OrderedDict()

    for cat_name, cat in info['Functions'].items():
        root_yaml[cat_name] = category_yaml(cat, info['Names'])

    utils.dump_yaml(root_yaml, open('functions.yaml', 'w'),
                    default_flow_style=False)


if __name__ == '__main__':
    main()
