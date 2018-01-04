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

from utils.load_function_rst import Functions
from utils.common import check_update
from utils.type_conv import type_from_proto

here = abspath(dirname(abspath(__file__)))
base = abspath(join(here, '../..'))

import code_generator_utils as utils


def info_to_list(info):
    '''Returns a list of (name, snake_name, [argument types as c++ type])'''
    items = []
    for name, item in info.items():
        items.append((name, item['snake_name'], [
            type_from_proto[v['type']]['cpp'] for v in item.get('arguments', {}).values()]))
    return items


def generate_init(function_info, function_types, solver_info, solver_types):
    # Create function list
    function_list = info_to_list(function_info)
    # Create solver list
    solver_list = info_to_list(solver_info)
    utils.generate_from_template(
        join(base, 'src/nbla/init.cpp.tmpl'),
        **utils.dict_filter(locals(), ['function_list', 'function_types', 'solver_list', 'solver_types']))


def generate_solver_python_intereface(solver_info):
    utils.generate_from_template(
        join(base, 'python/src/nnabla/solver.pyx.tmpl'), solver_info=solver_info)
    utils.generate_from_template(
        join(base, 'python/src/nnabla/solver.pxd.tmpl'), solver_info=solver_info)


def generate_function_types_template(function_info):
    from itertools import chain
    list(chain(*[cat.values for cat in function_info['Functions'].values()]))


def generate(function_info):
    solver_info = utils.load_yaml_ordered(
        open(join(here, 'solvers.yaml'), 'r'))
    function_info = utils.load_function_info(flatten=True)
    function_types = utils.load_yaml_ordered(open(
        join(here, 'function_types.yaml'), 'r'))
    solver_types = utils.load_yaml_ordered(open(
        join(here, 'solver_types.yaml'), 'r'))
    generate_init(function_info, function_types, solver_info, solver_types)
    generate_solver_python_intereface(solver_info)
