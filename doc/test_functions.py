#! /usr/bin/env python
from __future__ import print_function

import yaml
from collections import OrderedDict


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ils(indent_level):
    return ' ' * indent_level * 2


def print_yaml(y, indent_level=0):
    if isinstance(y, list):
        for i, v in enumerate(y):
            print(ils(indent_level) + '- %d' % i)
            print_yaml(v, indent_level + 1)
    elif isinstance(y, OrderedDict):
        for k, v in y.items():
            print(ils(indent_level) + k + ':')
            print_yaml(v, indent_level + 1)
    elif isinstance(y, str):
        print(ils(indent_level) + y.replace('\n', '\n' + ils(indent_level)))
    else:
        print(ils(indent_level) + str(y))


def main():

    print_yaml(ordered_load(open('functions.yaml', 'r')))


if __name__ == '__main__':
    main()
