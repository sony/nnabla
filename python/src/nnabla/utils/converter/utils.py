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
import json
from os.path import abspath, join, dirname
import os
import yaml
import zlib


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


def get_category_info_string():
    order = load_yaml_ordered(
        open(join(dirname(abspath(__file__)), 'function_order.yaml'), 'r'))
    string = open(join(dirname(abspath(__file__)),
                       'functions.yaml'), 'r').read()
    info = load_yaml_ordered(string)
    for cat, cat_info in info.items():
        for func, func_info in cat_info.items():
            if 'arguments' in func_info:
                for a, a_info in func_info['arguments'].items():
                    if 'default' in a_info:
                        a_info.pop('default')
                    if 'doc' in a_info:
                        a_info.pop('doc')
            for n, n_info in func_info['inputs'].items():
                if 'doc' in n_info:
                    n_info.pop('doc')
            for n, n_info in func_info['outputs'].items():
                if 'doc' in n_info:
                    n_info.pop('doc')
            func_info.pop('doc')

    for cat, cat_info in info.items():
        for func, func_info in cat_info.items():
            fmt = ''
            if 'arguments' in func_info:
                fmt = '_'
                for a, a_info in func_info['arguments'].items():
                    fmt += type_to_pack_format(a_info['type'])
            func_info['uniq_name'] = func + fmt
            func_info['id'] = order[func+fmt]
    return yaml.dump(info, default_flow_style=False)


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
