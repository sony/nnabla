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


from collections import OrderedDict
from os.path import abspath, join, dirname
import pickle
import yaml
import zlib

_nnabla_func_info = None
_onnx_func_info = None
_api_level_info = None


def func_set_init(func):
    def __ensure_load_pickle(*args):
        global _nnabla_func_info
        global _onnx_func_info
        global _api_level_info
        if _nnabla_func_info is None:
            with open(join(dirname(abspath(__file__)), 'functions.pkl'), 'rb') as f:
                data = pickle.load(f)
                _nnabla_func_info = data['nnabla_func_info']
                _onnx_func_info = data['onnx_func_info']
                _api_level_info = data['api_level_info']
        return func(*args)
    return __ensure_load_pickle


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


def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, represent_odict)


def load_yaml_ordered(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    '''
    Load function with keeping the order of dictionaries.
    '''
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


@func_set_init
def get_api_level_info():
    class FunctionInfo:
        def __init__(self, argument, func_id):
            self.argument = argument
            self.func_id = func_id

    class ApiLevelInfo:
        def __init__(self, d):
            self.api_level_dict = d
            self.api_level = 1
            self.function_dict = None

        def set_api_level(self, level):
            self.api_level = sorted(self.api_level_dict.keys()).pop()
            if level != -1 and level <= self.api_level:
                self.api_level = level
            self.function_dict = {uniq_name.split('_')[0]: FunctionInfo(uniq_name.split('_')[1], func_id)
                                  for uniq_name, func_id in self.api_level_dict[1].items()}
            for le in range(2, self.api_level + 1):
                f_dict = {uniq_name.split('_')[0]: FunctionInfo(uniq_name.split('_')[1], func_id)
                          for uniq_name, func_id in self.api_level_dict[le].items()}
                self.function_dict.update(f_dict)

        def get_current_level(self):
            return self.api_level

        def get_function_list(self):
            return self.function_dict.keys()

        def get_func_id(self, func_name):
            return self.function_dict[func_name].func_id

        def get_argument_code(self, func_name):
            argument_code = self.function_dict[func_name].argument
            if argument_code == 'Empty':
                argument_code = ''
            return argument_code

        def get_func_uniq_name(self, func_name):
            return func_name + '_' + self.function_dict[func_name].argument

    return ApiLevelInfo(_api_level_info)


@func_set_init
def get_category_info_string():
    header = '# Copyright (c) 2017 Sony Corporation. All Rights Reserved.\n' \
        + '#\n' \
        + '# Licensed under the Apache License, Version 2.0 (the "License");\n' \
        + '# you may not use this file except in compliance with the License.\n' \
        + '# You may obtain a copy of the License at\n' \
        + '#\n' \
        + '#     http://www.apache.org/licenses/LICENSE-2.0\n' \
        + '#\n' \
        + '# Unless required by applicable law or agreed to in writing, software\n' \
        + '# distributed under the License is distributed on an "AS IS" BASIS,\n' \
        + '# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n' \
        + '# See the License for the specific language governing permissions and\n' \
        + '# limitations under the License.\n' \
        + '#\n' \
        + '# DO NOT EDIT THIS FILE BY HAND\n' \
        + '# THIS FILE IS GENERATED FROM NNABLA.\n' \
        + '#\n' \
        + '\n'

    return header + yaml.dump(_nnabla_func_info, default_flow_style=False)


def get_category_info_version():
    return zlib.crc32(get_category_info_string().encode('utf-8')) & 0x7ffffff


def get_category_info():
    return load_yaml_ordered(get_category_info_string())


def get_function_info():
    functions = OrderedDict()
    for category_name, category in get_category_info().items():
        for function_name, function in category.items():
            functions[function_name] = function
    return functions


def select_executor(nnp, name=None):
    return nnp.protobuf.executor[0]


def search_network(nnp, name):
    for n in nnp.protobuf.network:
        if n.name == name:
            return n
    return None


def calc_shape_size(shape, batch_size):
    size = 1
    for d in shape.dim:
        if d < 0:
            d = batch_size
        size *= d
    return size


def func_set_import_nnp(nnp):
    network_name = nnp.protobuf.executor[0].network_name
    for _net in nnp.protobuf.network:
        if _net.name == network_name:
            return set([f.type for f in _net.function])


@func_set_init
def func_set_onnx_support():
    return set(_onnx_func_info.keys())


@func_set_init
def func_set_nncr_support():
    func_list = []
    for cat, cat_info in _nnabla_func_info.items():
        for func, func_info in cat_info.items():
            if 'c_runtime' in func_info and func_info['c_runtime'] == 'support':
                func_list.append(func)
    return set(func_list)


def func_set_import_config(config):
    func_list = []
    with open(config, "r") as f:
        for func_decl in f.readlines():
            func_decl = func_decl.strip()
            if func_decl.startswith(";"):
                continue
            else:
                func_list.append(func_decl.split(',')[0].strip())
    return set(func_list)


def func_dict_import_config(config):
    func_dict = {}
    legal_attr = ["FLOAT32", "FIXED16", "FIXED8", "ALL"]
    with open(config, "r") as f:
        for func_decl in f.readlines():
            func_decl = func_decl.strip()
            if func_decl.startswith(";"):
                continue
            else:
                func_decls = func_decl.split(',')
                func_dict[func_decls[0].strip()] = []
                if len(func_decls) == 1:
                    func_dict[func_decls[0].strip()].append('ALL')
                elif len(func_decls) > 1:
                    func_type = [t.strip().upper()
                                 for i, t in enumerate(func_decls) if i > 0]
                    if 'ALL' in func_type:
                        func_dict[func_decls[0].strip()].append('ALL')
                    else:
                        for t in func_type:
                            if t in legal_attr:
                                func_dict[func_decls[0].strip()].append(t)
    return func_dict


@func_set_init
def func_set_import_onnx_config(config):
    def handle_source_func_list(func):
        source_func_list.append(func)

    def handle_target_func_list(func, opset):
        # regular to declare standard
        if opset.startswith('opset_'):
            opset = opset[len('opset_'):]
        target_func_list.append("{}@{}".format(func, opset))

    def map_target_to_source(func_list):
        _func_list = []
        target_set = set(func_list)
        for nnabla_func, impl_funcs in _onnx_func_info.items():
            if set(impl_funcs) <= target_set:
                _func_list.append(nnabla_func)
        return _func_list

    source_func_list = []
    target_func_list = []
    with open(config, "r") as f:
        for func_decl in f.readlines():
            func_decl = func_decl.strip().split('@')
            if func_decl[0].startswith(";"):
                # comment out
                continue
            elif len(func_decl) == 1:
                handle_source_func_list(func_decl[0])
            elif len(func_decl) == 2:
                handle_target_func_list(func_decl[0], func_decl[1])
    if not source_func_list and not target_func_list:
        print("WARNING: function list seems empty!")
        return set()

    if target_func_list:
        func_list = map_target_to_source(target_func_list)
    else:
        func_list = source_func_list
    return set(func_list)


@func_set_init
def func_set_nnabla_support():
    func_list = []
    for cat, cat_info in _nnabla_func_info.items():
        for func, func_info in cat_info.items():
            func_list.append(func)
    return set(func_list)


@func_set_init
def func_set_onnx_output_target_list(func_set):
    target_list = []
    for func in func_set:
        target_list += _onnx_func_info[func]
    return set(target_list)


def func_set_import_onnx_opset(opset):
    opset = opset[len('opset_'):]
    target_func_list = []
    source_func_list = []
    for nnabla_func, impl_funcs in _onnx_func_info.items():
        for onnx_func in impl_funcs:
            _opset = onnx_func.split('@')[1]
            if _opset <= opset:
                target_func_list.append(onnx_func)
    for nnabla_func, impl_funcs in _onnx_func_info.items():
        if set(impl_funcs) <= set(target_func_list):
            source_func_list.append(nnabla_func)
    return set(source_func_list)


@func_set_init
def func_set_export_yaml(func_dict, yaml_file):
    for cat, cat_info in _nnabla_func_info.items():
        for func, func_info in cat_info.items():
            if func in func_dict.keys():
                func_info['func_type'] = func_dict[func]
            else:
                func_info['func_type'] = ['None']

    header = '# Copyright (c) 2017 Sony Corporation. All Rights Reserved.\n' \
        + '#\n' \
        + '# Licensed under the Apache License, Version 2.0 (the "License");\n' \
        + '# you may not use this file except in compliance with the License.\n' \
        + '# You may obtain a copy of the License at\n' \
        + '#\n' \
        + '#     http://www.apache.org/licenses/LICENSE-2.0\n' \
        + '#\n' \
        + '# Unless required by applicable law or agreed to in writing, software\n' \
        + '# distributed under the License is distributed on an "AS IS" BASIS,\n' \
        + '# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n' \
        + '# See the License for the specific language governing permissions and\n' \
        + '# limitations under the License.\n' \
        + '#\n' \
        + '# DO NOT EDIT THIS FILE BY HAND\n' \
        + '# THIS FILE IS GENERATED FROM NNABLA.\n' \
        + '#\n' \
        + '\n'

    with open(yaml_file, 'w') as f:
        f.write(header + yaml.dump(_nnabla_func_info, default_flow_style=False))


@func_set_init
def func_set_exporter_funcs_opset_yaml(func_set):
    if len(list(func_set)[0].split('@')) == 1:
        yaml_data = {}
        for nnabla_func, impl_funcs in _onnx_func_info.items():
            if nnabla_func in func_set:
                yaml_data[nnabla_func] = impl_funcs
        return yaml.dump(yaml_data, default_flow_style=False)
    else:
        return yaml.dump(list(func_set), default_flow_style=False)
