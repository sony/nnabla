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

import os
import sys
import collections

from .utils import read_nnp


def dump_protobuf(proto, prefix, depth):
    if depth >= 0 and len(prefix) >= depth:
        print('{} ...'.format(':'.join(prefix)))
        return
    for desc, field in proto.ListFields():
        if isinstance(field, (int, float, complex, str)):
            print('{}:{}: {}'.format(':'.join(prefix), desc.name, field))
        elif isinstance(field, collections.Iterable):
            print('{} has {} {}(s).'.format(
                ':'.join(prefix), len(field), desc.name))
            for n, f in enumerate(field[:10]):
                if isinstance(f, (int, float, complex, str)):
                    print('{}:{}[{}]: {}'.format(
                        ':'.join(prefix), desc.name, n, f))
                else:
                    if depth < 0 or depth > len(prefix)+1:
                        dump_protobuf(
                            f, prefix + ['{}[{}]'.format(desc.name, n)], depth)
        else:
            dump_protobuf(field, prefix + [desc.name], depth)


def dump_nnp(args, nnp):
    dump_protobuf(nnp.protobuf, [args.read_format], -1)
    return True


def dump_files(args, ifiles):
    nnp = read_nnp(args, ifiles)
    if nnp is not None:
        return dump_nnp(args, nnp)
    else:
        print('Read from [{}] failed.'.format(ifiles))
        return False
