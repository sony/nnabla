from collections import OrderedDict
import json
import os


def get_category_info_string():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'category_info.json')) as f:
        return f.read()


def get_category_info_version():
    return json.loads(get_category_info_string())['version']

def get_category_info():
    return json.loads(get_category_info_string())['categories']


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
