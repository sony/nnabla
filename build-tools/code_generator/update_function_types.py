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

from os.path import exists
import code_generator_utils as utils
from collections import OrderedDict


def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('path_types', type=str)
    p.add_argument('--default-type', type=str, default=None)
    args = p.parse_args()
    return args


def main():
    args = get_args()
    func_info = utils.load_function_info(flatten=True)
    if exists(args.path_types):
        func_types = utils.load_yaml_ordered(open(args.path_types, 'r'))
    else:
        func_types = OrderedDict()
    for name, func in func_info.items():
        if name in func_types:
            continue
        print("Processing %s..." % name)
        types = OrderedDict()
        if args.default_type is not None:
            types[args.default_type] = [args.default_type]
        func_types[name] = types
    utils.dump_yaml(func_types, open(args.path_types, 'w'))


if __name__ == '__main__':
    main()
