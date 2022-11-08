# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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


import pickle
import zlib
from collections import OrderedDict
from os.path import abspath, join, dirname, expanduser
import glob
import os
import yaml
import certifi
import ssl
from nnabla.utils import nnabla_pb2
from nnabla.logger import logger


_nnabla_func_info = None
_nnabla_func_info_old = {}
_nnabla_func_info_cur = {}
_onnx_func_info = None
_api_level_info = None
_nnabla_func_info_versions = {}


def create_context():
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(certifi.where())
    return context


ssl._create_default_https_context = create_context


def load_yaml_ordered(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def load_function_info(yaml_file):
    with open(yaml_file, 'r') as f:
        ret = load_yaml_ordered(f)
    ret2 = OrderedDict()
    for _, cat in ret.items():
        for func_name, func in cat.items():
            ret2[func_name] = func
    return ret2


def download_func_info_yaml(nnp_version):
    import urllib.request
    cache_dir = os.path.join(expanduser("~"), ".nnabla")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    yaml_file = os.path.join(cache_dir, f"function_info_{nnp_version}.yaml")
    try:
        logger.info(
            f"Downloading function information for nnabla v{nnp_version}, it will take a few time...")
        url = f"https://raw.githubusercontent.com/sony/nnabla/v{nnp_version}/build-tools/code_generator/functions.yaml"
        logger.info(f"Retrieve from {url} ...")
        urllib.request.urlretrieve(url, yaml_file)
        logger.info(f"Function information is downloaded to: {yaml_file}")
    except Exception as e:
        print(e)


def func_set_nnabla_version_decorate(nnp, nnp_version):
    import re

    pattern = re.compile(r'(?<!^)(?=[A-Z])')

    def convert_from_camel_to_snake(name):
        '''Get function name from camel to snake
        '''
        name = pattern.sub('_', name).lower()
        return name

    if nnp_version is None:
        return nnp

    network_name = nnp.protobuf.executor[0].network_name
    exec_network = None
    for n in nnp.protobuf.network:
        if n.name == network_name:
            exec_network = n
            for f in n.function:
                no_param = False
                try:
                    # Some snake name of function type does not follow the conversion rule
                    # between camel and snake.
                    eval(f"f.{convert_from_camel_to_snake(f.type) +'_param'}")
                    f_param_name = f"f.{convert_from_camel_to_snake(f.type) +'_param'}"
                    param_name = f"{convert_from_camel_to_snake(f.type) +'_param'}"
                except AttributeError:
                    try:
                        eval(f"f.{f.type.lower() + '_param'}")
                        f_param_name = f"f.{f.type.lower() + '_param'}"
                        param_name = f"{f.type.lower() + '_param'}"
                    except AttributeError:
                        no_param = True

                old_info = _nnabla_func_info_old[f.type]
                cur_info = _nnabla_func_info_cur[f.type]

                if no_param:
                    if 'arguments' in old_info:
                        raise ValueError(
                            f"{f} is not supported for arguments imcompatible change.")
                    else:
                        continue

                for arg_name, arg in cur_info['arguments'].items():
                    if 'arguments' not in old_info:
                        def_arg = cur_info['arguments'][arg_name]['default']
                        arg = str(
                            eval(f"{f_param_name}.{arg_name}"))
                        if def_arg != arg:
                            raise ValueError(
                                f"{f} only support default argument: {arg_name}={def_arg}, but actual {arg_name}={arg}!")
                        exec(f"f.ClearField('{param_name}')")
                    elif arg_name not in old_info['arguments']:
                        exec(
                            f"{f_param_name}.ClearField('{arg_name}')")

                if 'arguments' in old_info:
                    for arg_name, arg in old_info['arguments'].items():
                        if 'arguments' not in cur_info:
                            raise ValueError(
                                f"{f} is not supported for arguments imcompatible change.")
                        if arg_name not in cur_info['arguments']:
                            # default MUST exist
                            if isinstance(arg, dict):
                                assert 'default' in arg, "We assume 'default' value is set for this argument."
                                exec(
                                    f"{f_param_name} + '.' + {arg_name} = {arg['default']}")

    if exec_network is not None:
        proto = nnabla_pb2.NNablaProtoBuf()
        network = proto.network.add()
        network.CopyFrom(exec_network)
        executor = proto.executor.add()
        executor.CopyFrom(nnp.protobuf.executor[0])
        for p in nnp.protobuf.parameter:
            param = proto.parameter.add()
            param.CopyFrom(p)
        nnp.protobuf = proto

    return nnp


def extract_version_from_filename(yaml_file):
    basename = os.path.basename(yaml_file)
    basename = os.path.splitext(basename)[0]
    splits = basename.split('_')
    if len(splits) < 3:
        return None
    return splits[2]


def load_yaml_from_cache_dir():
    yaml_files = os.path.join(expanduser(
        "~"), ".nnabla", "function_info_*.yaml")
    func_yaml_files = glob.glob(yaml_files)
    for yaml_file in func_yaml_files:
        version = extract_version_from_filename(yaml_file)
        if version is None:
            continue
        _nnabla_func_info_versions[version] = load_function_info(yaml_file)


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
            load_yaml_from_cache_dir()
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
    header = '# Copyright 2021 Sony Group Corporation. \n' \
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
def func_set_get_from_repo(nnp_version):
    nnabla_func_info = _nnabla_func_info_versions.get(nnp_version, None)
    if nnabla_func_info is None:
        download_func_info_yaml(nnp_version)
        load_yaml_from_cache_dir()
        nnabla_func_info = _nnabla_func_info_versions.get(nnp_version, None)

    if nnabla_func_info is None:
        raise ValueError(f"nnabla v{nnp_version} does not exist?")

    global _nnabla_func_info_old
    _nnabla_func_info_old = nnabla_func_info

    return set(list(nnabla_func_info.keys()))


def _func_set_nnabla_unsupport():
    """
    This function performs compatibility check
    """
    cur_set = func_set_nnabla_support()
    unsupported = set()
    for f in cur_set:
        cur_info = _nnabla_func_info_cur[f]
        if f not in _nnabla_func_info_old:
            unsupported.add(f)
            continue

        old_info = _nnabla_func_info_old[f]

        # check input compatibility
        for inp_name, inp_dict in cur_info['inputs'].items():
            if inp_name not in old_info['inputs']:
                unsupported.add(f)
                break
            old_inp_dict = old_info['inputs'][inp_name]
            for attr_name, attr in inp_dict.items():
                if attr_name == 'doc':
                    continue
                # Loose the check for `optional` change
                # if attr_name not in old_inp_dict:
                #     unsupported.add(f)
                #     break
                if attr_name in old_inp_dict:
                    if old_inp_dict[attr_name] != attr:
                        unsupported.add(f)
                        break
            if f in unsupported:
                break

        # check arguments compability
        # if argment in old, not in new, set to old default value
        #     if no default value, add to unsupported set
        # if argment in new, not in old, remove it
        if f not in unsupported:
            if 'arguments' in cur_info and 'arguments' in old_info:
                for arg_name, arg_dict in old_info['arguments'].items():
                    if arg_name not in cur_info['arguments']:
                        if 'default' not in arg_dict:
                            unsupported.add(f)
                            break

        if f not in unsupported:
            if len(cur_info['outputs']) != len(old_info['outputs']):
                unsupported.add(f)
    return unsupported


@func_set_init
def func_set_nnabla_support(version_spec=None):
    if version_spec is None:
        func_list = []
        for cat, cat_info in _nnabla_func_info.items():
            for func, func_info in cat_info.items():
                func_list.append(func)
        return set(func_list)
    else:
        nnp, nnp_version = version_spec
        # load func_info to _nnabla_func_info_cur
        for cat, cat_info in _nnabla_func_info.items():
            for func, func_info in cat_info.items():
                _nnabla_func_info_cur[func] = func_info
        old_set = func_set_get_from_repo(nnp_version)
        unsupported = _func_set_nnabla_unsupport()
        # logger.info(f"The following functions are not supported in version: {nnp_version}:")
        # for f in unsupported:
        #     logger.info(f"{f}")
        old_set -= unsupported
        nnp_set = func_set_import_nnp(nnp)
        if nnp_set & old_set != nnp_set:
            unsupported_functions = nnp_set - old_set
            logger.error(
                f"The following functions are not supported in nnabla v{nnp_version} but appear in .nnp file.")
            for f in unsupported_functions:
                logger.error(f"{f}")
            raise ValueError(
                f"nnp file contains unsupported functions by nnabla version: {nnp_version}.")
        return old_set


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

    header = '# Copyright 2021 Sony Group Corporation.\n' \
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
