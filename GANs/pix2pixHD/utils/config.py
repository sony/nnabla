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


class Config(object):
    '''A helper class which makes handling config yaml easier in Python.

    '''

    def __init__(self, filename=None, item=None):
        assert filename is None or item is None
        assert filename is not None or item is not None
        if filename is not None:

            with open(filename, 'r') as fd:
                self._item = yaml.safe_load(fd)
        else:
            self._item = item

    def __str__(self):
        return self._item.__str__()

    def __repr__(self):
        return 'Config({})'.format(self._item.__repr__())

    def save_(self, filename):
        with open(filename, 'w') as fd:
            yaml.dump(self._item, fd, default_flow_style=None)

    def get_items_(self, *keys):
        return {k: self[k] for k in keys}

    def __getattr__(self, key):
        if key not in self._item:
            return None
        item = self._item[key]
        if isinstance(item, dict):
            return Config(None, item)
        return item

    def __getitem__(self, key):
        return self.__getattr__(key)


def makedirs(p):
    if os.path.isdir(p):
        return
    os.makedirs(p)


def read_config(filename, backup=True):
    cfg = Config(filename)

    # Create save path
    p = cfg.train.save_path
    makedirs(p)

    # Backup config yaml at save path
    cfg.save_(os.path.join(p, 'config.yaml'))
    return cfg
