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
import sys
from collections import OrderedDict
from os.path import abspath, dirname, join, exists

here = abspath(dirname(abspath(__file__)))
base = abspath(join(here, '../..'))

import code_generator_utils as utils
import yaml

def type_to_pack_format(typestring):
    fmt = None
    if typestring == 'bool':
        fmt = 'B'
    elif typestring == 'double' or typestring == 'float':
        fmt = 'f'
    elif typestring == 'int64':
        fmt = 'i'
    elif typestring == 'repeated int64' or typestring == 'Shape':
        fmt = 'iI'
    elif typestring == 'string':
        fmt = 'i'
    elif typestring == 'Communicator':
        fmt = 'C'
    return fmt

def generate_cpp_utils(function_info):
    function_list = utils.info_to_list(function_info)
    utils.generate_from_template(
        join(base, 'src/nbla_utils/nnp_impl_create_function.cpp.tmpl'), function_info=function_info, function_list=function_list)


def generate_proto(function_info, solver_info):
    utils.generate_from_template(
        join(base, 'src/nbla/proto/nnabla.proto.tmpl'), function_info=function_info, solver_info=solver_info)


def generate_python_utils(function_info):
    utils.generate_from_template(
        join(base, 'python/src/nnabla/utils/load_function.py.tmpl'), function_info=function_info)
    utils.generate_from_template(
        join(base, 'python/src/nnabla/utils/save_function.py.tmpl'), function_info=function_info)


def generate_function_python_interface(function_info):
    utils.generate_from_template(
        join(base, 'python/src/nnabla/function.pyx.tmpl'), function_info=function_info)
    utils.generate_from_template(
        join(base, 'python/src/nnabla/function.pxd.tmpl'), function_info=function_info)
    utils.generate_from_template(
        join(base, 'python/src/nnabla/function_bases.py.tmpl'), function_info=function_info)


def generate_solver_python_interface(solver_info):
    utils.generate_from_template(
        join(base, 'python/src/nnabla/solver.pyx.tmpl'), solver_info=solver_info)
    utils.generate_from_template(
        join(base, 'python/src/nnabla/solver.pxd.tmpl'), solver_info=solver_info)

def update_function_order_in_functsions_yaml():
    d = utils.load_yaml_ordered(open(join(here, 'functions.yaml'), 'r'))

    order_info_by_id = {}
    order_info = OrderedDict()

    duplicated = {}
    missing = {}

    for cat_name, cat_info in d.items():
        for func_name, func_info in d[cat_name].items():
            order_info[func_name] = OrderedDict()

            default_full_name = func_name
            default_arg = ''
            if 'arguments' in func_info:
                for arg, arg_info in func_info['arguments'].items():
                    default_arg += type_to_pack_format(arg_info['type'])
            if default_arg == '':
                default_arg = 'Empty'
            else:
                default_full_name = func_name + '_' + default_arg

            if 'function_ids' in func_info and func_info['function_ids'] is not None:
                for func_arg, func_id in func_info['function_ids'].items():
                    full_name = func_name
                    if func_arg != 'Empty':
                        full_name = func_name + '_' + func_arg

                    if func_id in order_info_by_id:
                        if func_id not in duplicated:
                            duplicated[func_id] = [order_info_by_id[func_id]]
                        duplicated[func_id].append(full_name)

                    order_info_by_id[func_id] = full_name
                    order_info[func_name][full_name] = func_id
                if default_full_name not in order_info[func_name]:
                    if cat_name not in missing:
                        missing[cat_name] = {}
                    if func_name not in missing[cat_name]:
                        missing[cat_name][func_name] = []
                    missing[cat_name][func_name].append(default_arg)
            else:
                if cat_name not in missing:
                    missing[cat_name] = {}
                if func_name not in missing[cat_name]:
                    missing[cat_name][func_name] = []
                missing[cat_name][func_name].append(default_arg)

            if 'c_runtime' not in func_info:
                func_info['c_runtime'] = 'not support'

    current_id = sorted(order_info_by_id.keys()).pop() + 1
    for cat_name in missing:
        for func_name in missing[cat_name]:
            for arg in missing[cat_name][func_name]:
                if 'function_ids' not in d[cat_name][func_name] or d[cat_name][func_name]['function_ids'] is None:
                    d[cat_name][func_name]['function_ids'] = OrderedDict()
                d[cat_name][func_name]['function_ids'][arg] = current_id
                current_id += 1

    if len(duplicated):
        print('')
        print('############################################## Errors in functions.yaml(START)')
        for func_id, functions in duplicated.items():
            if len(functions) > 1:
                print('ID {} duplicated between {}.'.format(func_id, functions))
        print('Correct ID in "build-tools/code_generator/functions.yaml" manually.')
        print('############################################## Errors in functions.yaml(END)')
        print('')
        import sys
        sys.exit(-1)

    utils.dump_yaml(d, open(join(here, 'functions.yaml'), 'w'), default_flow_style=False, width=80)

def generate_functions_pkl():
    import pickle
    yaml_data = {}
    d = utils.load_yaml_ordered(open(join(here, 'functions.yaml'), 'r'))

    for cat_name, cat_info in d.items():
        for func_name, func_info in d[cat_name].items():
            if 'doc' in func_info:
                del func_info['doc']
            for a in ['inputs', 'arguments', 'outputs']:
                if a in func_info:
                    for b in func_info[a]:
                        if 'doc' in func_info[a][b]:
                            del func_info[a][b]['doc']
            fmt = ''
            if 'arguments' in func_info:
                fmt = '_'
                for a, a_info in func_info['arguments'].items():
                    fmt += type_to_pack_format(a_info['type'])
            func_info['uniq_name'] = func_name + fmt
            func_info['id'] = list(func_info['function_ids'].items()).pop()[1]
    yaml_data['nnabla_func_info'] = d

    o = utils.load_yaml_ordered(open(join(base, 'python/test/utils/conversion/exporter_funcs_opset.yaml'), 'r'))
    yaml_data['onnx_func_info'] = {}
    for func, func_info in o.items():
        if 'Not implemented' in func_info:
            continue
        else:
            yaml_data['onnx_func_info'][func] = func_info

    with open(join(base, 'python/src/nnabla/utils/converter/functions.pkl'), 'wb') as f:
        pickle.dump(yaml_data, f, 2)

def generate_function_cpp_interface(function_info):
    function_list = utils.info_to_list(function_info)
    utils.generate_from_template(
        join(base, 'include/nbla/functions.hpp.tmpl'), function_info=function_info, function_list=function_list)
    utils.generate_from_template(
        join(base, 'src/nbla/functions.cpp.tmpl'), function_info=function_info, function_list=function_list)

def generate_backward_function_mapping(function_info):
    function_list = utils.info_to_list(function_info)
    utils.generate_from_template(
        join(base, 'python/src/nnabla/backward_functions.py.tmpl'),
        function_info=function_info, function_list=function_list)

def generate():
    version = sys.argv[1]
    update_function_order_in_functsions_yaml()
    generate_functions_pkl()

    function_info = utils.load_function_info(flatten=True)
    solver_info = utils.load_solver_info()
    function_types = utils.load_yaml_ordered(open(join(here, 'function_types.yaml'), 'r'))
    solver_types = utils.load_yaml_ordered(open(join(here, 'solver_types.yaml'), 'r'))
    utils.generate_init(function_info, function_types, solver_info, solver_types)
    utils.generate_function_types(function_info, function_types)
    utils.generate_solver_types(solver_info, solver_types)
    utils.generate_version(join(base, 'python/src/nnabla/_version.py.tmpl'), base, version=version)
    utils.generate_version(join(base, 'src/nbla/version.cpp.tmpl'), base, version=version)
    utils.generate_version(join(base, 'doc/requirements.txt.tmpl'), base, version=version)
    generate_solver_python_interface(solver_info)
    generate_function_python_interface(function_info)
    generate_python_utils(function_info)
    generate_proto(function_info, solver_info)
    generate_cpp_utils(function_info)
    generate_function_cpp_interface(function_info)
    generate_backward_function_mapping(function_info)

    # Generate function skeletons if new ones are added to functions.yaml and function_types.yaml.
    utils.generate_skeleton_function_impl(
        function_info, function_types)
    func_header_template = join(
        base,
        'include/nbla/function/function_impl.hpp.tmpl')
    utils.generate_skeleton_function_impl(
        function_info, function_types,
        template=func_header_template, output_format='%s.hpp')

    # Generate backward function skeletons if new ones are added to functions.yaml
    utils.generate_skeleton_backward_function_impl(function_info,
                                                   join(base,
                                                        'python/src/nnabla/backward_function_class.py.tmpl'),
                                                   join(base,
                                                        'python/src/nnabla/backward_function'))

    # TODO: solver skeleton generation


if __name__ == '__main__':
    generate()
