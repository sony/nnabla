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

import nnabla as nn


def insert_parent_name(name, params):
    ret = OrderedDict()
    for k, v in params.items():
        ret['/'.join(('@' + name, k))] = v
    return ret


class ParamMemo(object):
    def __init__(self):
        self._memo = set()

    def filter_and_update(self, params):
        memo = self._memo
        ret = OrderedDict()
        for k, v in params.items():
            if v in memo:
                continue
            ret[k] = v
            memo.add(v)
        return ret


class Module(object):

    def __repr__(self):
        return f'{self.__class__.__name__}'

    @property
    def training(self):
        if '_training' in self.__dict__:
            return self._training
        self.__dict__['_training'] = True
        return self._training

    @training.setter
    def training(self, b):
        self.__dict__['_training'] = b
        for name, module in self.submodules.items():
            module.training = b

    @property
    def parameter_scope(self):
        if '_parameter_scope' in self.__dict__:
            return self._parameter_scope
        self.__dict__['_parameter_scope'] = OrderedDict()
        return self._parameter_scope

    @property
    def submodules(self):
        if '_submodules' in self.__dict__:
            return self._submodules
        self.__dict__['_submodules'] = OrderedDict()
        return self._submodules

    def get_parameters(self, recursive=True, grad_only=False, memo=None):
        params = OrderedDict()
        if memo is None:
            memo = ParamMemo()
        if recursive:
            for name, module in self.submodules.items():
                params.update(
                    insert_parent_name(
                        name, module.get_parameters(
                            recursive=recursive, grad_only=grad_only,
                            memo=memo)))
        with nn.parameter_scope('', self.parameter_scope):
            found_params = nn.get_parameters(grad_only=grad_only)
            filtered_params = memo.filter_and_update(found_params)
            params.update(filtered_params)
        return params

    def set_parameter(self, key, param, raise_if_missing=False):
        if key.startswith('@'):
            # Recursively set parameters
            pos = key.find('/')
            if pos < 0 or pos == len(key) - 1:
                raise ValueError(
                    f'Invalid parameter key {key}.'
                    ' A module parameter scope represented'
                    ' as `@name` must be followed by `/`.')
            module_name, subkey = key[1:pos], key[pos + 1:]
            if module_name in self.submodules.keys():
                self.submodules[module_name].set_parameter(subkey, param)
            elif raise_if_missing:
                raise ValueError(
                    f'A child module {module_name[1:]} cannot be found in'
                    '{this}. This error is raised because `raise_if_missing`'
                    'is specified as True. Please turn off if you allow it.')
            return

        # Set parameters
        with nn.parameter_scope('', self.parameter_scope):
            nn.parameter.set_parameter(key, param)
            # nn.logger.info(f'`({key}` loaded.)')

    def set_parameters(self, params, raise_if_missing=False):
        for key, param in params.items():
            self.set_parameter(key, param, raise_if_missing=raise_if_missing)

    def load_parameters(self, path, raise_if_missing=False):
        r"""Loads parameters from a file with the specified format.

        Args:
            path (str): The path to file.
            raise_if_missing (bool, optional): Raise exception if some
                parameters are missing. Defaults to `False`.
        """
        with nn.parameter_scope('', OrderedDict()):
            nn.load_parameters(path)
            params = nn.get_parameters(grad_only=False)
        self.set_parameters(params, raise_if_missing=raise_if_missing)

    def save_parameters(self, path, grad_only=False):
        r"""Saves the parameters to a file.

        Args:
            path (str): Path to file.
            grad_only (bool, optional): If `need_grad=True` is required for
                parameters which will be saved. Defaults to False.
        """
        params = self.get_parameters(grad_only=grad_only)
        with nn.parameter_scope('', OrderedDict()):
            nn.save_parameters(path, params)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.submodules[name] = value
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in self.submodules:
            return self.submodules[name]
        return self.__dict__[name]

    def __call__(self, *args, **kwargs):
        with nn.parameter_scope('', self.parameter_scope):
            return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        raise NotImplementedError('call(*av, **kw) must be implemented.')
