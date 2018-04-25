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
import os
import yaml
import zlib


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
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'functions.yaml')) as f:
        return f.read()


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
