# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
import yaml
import re

from .misc import AttrDict, makedirs


# Resolve scientific notation: https://github.com/yaml/pyyaml/pull/174/files
yaml.resolver.Resolver.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$''', re.X),
    list('-+0123456789.'))


def read_yaml(filepath):
    with open(filepath, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return AttrDict(data)


def write_yaml(filepath, obj):
    dirname = os.path.dirname(filepath)

    makedirs(dirname)

    with open(filepath, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False)
