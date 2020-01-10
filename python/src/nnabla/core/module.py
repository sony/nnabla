# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

import re
import nnabla as nn
from collections import OrderedDict
from nnabla.core.graph_def import ProtoVariable, module_scope, current_graph_builder

# TODO:
#   - The following submodule has not supported yet, for example:
#         class ResUnit(Module):
#             def __init__(self, channels, stride=1, skip_by_conv=True):
#                 self.conv = [ConvBn(c, 1, 1,act=lambda x: F.relu(x, inplace=True)) for c in [4, 16, 32]]
#                 ...
#     same as dict type.


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
    """Module is a construction block of a computation model.

    Example:
        User may construct his model by derived from this class. Like:
        .. code-block:: python

            import nnabla as nn

            class ConvBn(Module):
                def __init__(self, outmaps, kernel=1, stride=1, act=None):
                    self.outmaps = outmaps
                    self.kernel = kernel
                    self.stride = stride
                    self.act = act

                def call(self, x, training=True):
                    kernel = complete_dims(self.kernel, 2)
                    pad = get_conv_same_pad(kernel)
                    stride = complete_dims(self.stride, 2)
                    h = PF.convolution(x, self.outmaps, kernel,
                                       pad, stride, with_bias=False)
                    h = PF.batch_normalization(h, batch_stat=training)
                    if self.act is None:
                        return h
                    return self.act(h)

            class ResUnit(Module):
                def __init__(self, channels, stride=1, skip_by_conv=True):
                    self.conv1 = ConvBn(channels // 4, 1, 1,
                                        act=lambda x: F.relu(x, inplace=True))
                    self.conv2 = ConvBn(channels // 4, 3, stride,
                                        act=lambda x: F.relu(x, inplace=True))
                    self.conv3 = ConvBn(channels, 1)
                    self.skip_by_conv = skip_by_conv
                    self.skip = ConvBn(channels, 1, stride)

                def call(self, x, training=True):

                    h = self.conv1(x)
                    h = self.conv2(h)
                    h = self.conv3(h)

                    s = x
                    if self.skip_by_conv:
                        s = self.skip(s)
                    h = F.relu(F.add2(h, s, inplace=True), inplace=True)
                    return h


        To use this model, user may use the following code:

        .. code-block:: python

            res_unit = ResUnit(1024)
            x = nn.Variable((64, 3, 32, 32))
            x.d = np.random.random(x.shape)
            y = res_unit(x)
            y.forward(clear_buffer=True)

    """

    def __repr__(self):
        return self.__class__.__name__

    @property
    def training(self):
        if '_training' in self.__dict__:
            return self._training
        self.__dict__['_training'] = None
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

    def get_scope_name(self):
        scope_name = ''
        for name, module in self.submodules.items():
            scope_name = '/'.join(('@' + name, module.get_scope_name()))
            break
        return scope_name.strip('/')

    @property
    def submodules(self):
        if '_submodules' in self.__dict__:
            return self._submodules
        self.__dict__['_submodules'] = OrderedDict()
        return self._submodules

    def get_functions(self, recursive=True):
        def _get_recursive(values, recursive):
            for name, module in self.submodules.items():
                values.update(
                    insert_parent_name(
                        name,
                        module.get_functions(recursive=recursive)))
        functions = OrderedDict(self.functions)
        if recursive:
            _get_recursive(functions, recursive)
        return functions

    def get_parameters(self, recursive=True, grad_only=False, memo=None):
        '''
        '''
        params = OrderedDict()
        if memo is None:
            memo = ParamMemo()
        if recursive:
            for name, module in self.submodules.items():
                params.update(
                    insert_parent_name(
                        name,
                        module.get_parameters(recursive=recursive, grad_only=grad_only, memo=memo)))
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
                    'Invalid parameter key {}.'
                    ' A module parameter scope represented'
                    ' as `@name` must be followed by `/`.'.format(key))
            module_name, subkey = key[1:pos], key[pos + 1:]
            if module_name in self.submodules.keys():
                self.submodules[module_name].set_parameter(subkey, param)
            elif raise_if_missing:
                raise ValueError(
                    'A child module {} cannot be found in {}. '
                    'This error is raised because `raise_if_missing` is specified '
                    'as True. Please turn off if you allow it.'.format(module_name[1:], self))
            return

        # Set parameters
        with nn.parameter_scope('', self.parameter_scope):
            nn.parameter.set_parameter(key, param)

    def set_parameters(self, params, raise_if_missing=False):
        for key, param in params.items():
            self.set_parameter(key, param, raise_if_missing=raise_if_missing)

    def __len__(self):
        return len(self.submodules)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.submodules[name] = value
            value.__dict__['_name'] = name
            value.__dict__['_parent'] = self
            return
        self.__dict__[name] = value

    def __getattr__(self, name):
        if name in self.submodules:
            return self.submodules[name]
        return self.__dict__[name]

    @property
    def name(self):
        if '_name' in self.__dict__:
            return self.__dict__['_name']
        name = self.__class__.__name__
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return name

    @property
    def parent(self):
        return self.__dict__.get('_parent', None)

    def is_root(self):
        return self.parent is None

    def get_path_name(self):
        if self.is_root():
            return '@' + self.name
        else:
            parent_name = self.parent.get_path_name()
            if parent_name:
                return '/'.join([parent_name, '@' + self.name])
            else:
                return '@' + self.name

    def __call__(self, *args, **kwargs):
        for i in args:
            if isinstance(i, ProtoVariable):
                with module_scope(current_graph_builder(), self):
                    with nn.parameter_scope('', self.parameter_scope):
                        ret = self.call(*args, **kwargs)
                        return ret
        with nn.parameter_scope('', self.parameter_scope):
            ret = self.call(*args, **kwargs)
            return ret

    def call(self, *args, **kwargs):
        raise NotImplementedError('call(*av, **kw) must be implemented.')
