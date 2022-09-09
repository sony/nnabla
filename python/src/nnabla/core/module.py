# Copyright 2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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
from collections import OrderedDict
from functools import wraps

import nnabla as nn
from nnabla.core.graph_def import module_scope, current_graph_builder


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


class MetaClass(type):
    @staticmethod
    def method_wrapper(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            with module_scope(current_graph_builder(), self):
                with nn.parameter_scope('', self.parameter_scope):
                    return method(self, *args, **kwargs)
        return wrapped

    def __new__(meta, classname, bases, class_dict):
        if not bases:
            # Skip applying method_wrapper if it is a base class.
            return type.__new__(meta, classname, bases, class_dict)

        new_class_dict = {}
        for name, attr in class_dict.items():
            if callable(attr):
                attr = MetaClass.method_wrapper(attr)
            new_class_dict[name] = attr
        return type.__new__(meta, classname, bases, new_class_dict)


class Module(metaclass=MetaClass):
    """Module is a construction block of a computation model. Modules normally
    are constructed by lower level operators or other Modules, thus, nesting
    them in a tree-like structure may construct a more complex computation
    model.

    Example:
        User may construct his model by derived from this class. Like:

        .. code-block:: python

            import nnabla as nn
            import nnabla.parametric_functions as PF
            import nnabla.functions as F

            class ConvBn(nn.Module):
                def __init__(self, outmaps, kernel=1, stride=1, act=None):
                    self.outmaps = outmaps
                    self.kernel = kernel
                    self.stride = stride
                    self.act = act

                def call(self, x, training=True):
                    kernel = (self.kernel, self.kernel)
                    pad = (self.kernel // 2, self.kernel // 2)
                    stride = (self.stride, self.stride)
                    h = PF.convolution(x, self.outmaps, kernel,
                                       pad, stride, with_bias=False)
                    h = PF.batch_normalization(h, batch_stat=training)
                    if self.act is None:
                        return h
                    return self.act(h)


            class ResUnit(nn.Module):
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


        To use this model, user may do like the following code:

        .. code-block:: python

            res_unit = ResUnit(1024)
            x = nn.Variable((64, 3, 32, 32))
            x.d = np.random.random(x.shape)
            y = res_unit(x)
            y.forward(clear_buffer=True)


        For working with dynamic network, user may do like the following:

        .. code-block:: python

            res_unit = ResUnit(1024)
            with nn.auto_forward():
                x = nn.Variable.from_numpy_array(np.random.random((1, 3, 32, 32)))
                y = res_unit(x)
                print(y.d)


        For training, please set the parameters in module scope to optimizer. For example,

        .. code-block:: python

           import nnabla.solvers as S

           resnet = ResNet(18)
           loss = resnet(x, y_)

           solver = S.Sgd(lr=1e-3)
           solver.set_parameters(resnet.get_parameters())

           for _ in range(max_iter):
               x.d, y_.d = data.next()
               loss.forward()
               solver.zero_grad()
               loss.backward()
               solver.weight_decay(1e-5)
               solver.update()

        In this example, we supposed ResNet is a derived class of Module, x, y_ is :class:`~nn.Variable`,
        ``data`` is an instance of :class:`~DataIterator`, supposed it has already been attached to a DataSet.

        Note:
            From this example, we knew that model parameters are owned by model. Here it is variable `resnet`. These
            parameters will be referred when network is forward or backward or solve.update(). Hence, it is necessary
            to keep this module instance from being unexpectedly released, to ensure forward() or backward() can refer
            to these variables.
    """

    _recursive_attrs = ["training"]

    def __repr__(self):
        return self.__class__.__name__

    @property
    def training(self):
        """Return a bool value which indicates whether current Module is in
        training state or not. A module may be set to training state or not,
        so that the computation graph created from this module can be changed
        according to this state. For example,

        .. code-block:: python

            class ConvBN(Module):
                ...
                def call(self, x):
                    h = self.conv1(x)
                    if self.training:
                        h = self.drop_out(h)
                    h = F.relu(h, inplace=True)
                    return h

            conv_bn = ConvBN()
            conv_bn.training = True
            train_y = conv_bn(x)

            conv_bn.training = False
            eval_y = conv_bn(x)

        Returns:
            bool:
               which indicates whether current Module is in training state.

        """
        if 'training' in self.__dict__:
            return self.__dict__['training']
        self.__dict__['training'] = True
        return self.training

    @training.setter
    def training(self, b):
        """Set current Module whether is in training state or not.

        .. code-block:: python

            class ConvBN(Module):
                ...
                def call(self, x):
                    h = self.conv1(x)
                    if self.training:
                        h = self.drop_out(h)
                    h = F.relu(h, inplace=True)
                    return h

            conv_bn = ConvBN()
            conv_bn.training = True
            train_y = conv_bn(x)

            conv_bn.training = False
            eval_y = conv_bn(x)
        """
        # This is not called since __setattr__ takes off its control.
        # Keep it only for API document
        pass

    @property
    def parameter_scope(self):
        """ A module has its owned parameter_scope, which can avoid to pollute global parameter name space.
        User may obtain the parameter_scope of a module by this property.

        Returns:
            OrderedDict:
                The parameter scope of current Module.
        """
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

    def get_parameters(self, recursive=True, grad_only=False, memo=None):
        """Obtain an OrderedDict object of all parameters in current Module.

        For example,

        .. code-block:: python

            x = nn.Variable.from_numpy_array((np.random.random((8, 32, 256, 256))))
            conv_bn = ConvBn(2)
            y = conv_bn(x)

            params = conv_bn.get_parameters()
            for parameter_name, parameter_value in params.items():
                print("{}:{}".format(parameter_name, parameter_value.shape))

        The output looks like:

        .. code-block:: none

            conv/W:(2, 32, 1, 1)
            bn/beta:(1, 2, 1, 1)
            bn/gamma:(1, 2, 1, 1)
            bn/mean:(1, 2, 1, 1)
            bn/var:(1, 2, 1, 1)

        Notice that the parameter name looks like a filepath, with splash separated
        nested scope name. In addition, module name default is used with a prefix ``@``.

        Args:
            recursive (bool, optional, default=True):
                Whether obtain the parameters of current module's submodules. Default is True.
            grad_only (bool, optional, default=False):
                Whether only obtain the grad. Default is False.

        Returns:
            OrderedDict:
                Flattened parameter's name-value pairs of current Module.
        """
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

    def load_parameters(self, path, extension=".h5", raise_if_missing=True):
        """Load parameters from a file into this module.

        Args:
            path: str or file-like object

        """
        scope = OrderedDict()
        with nn.parameter_scope('', scope):
            nn.load_parameters(path, extension=extension)
            params = nn.get_parameters(grad_only=False)
        self.set_parameters(params, raise_if_missing=raise_if_missing)

    def save_parameters(self, path, extension=".h5"):
        """Save parameters of this module to a file.

        Args:
            path: str or file-like object
        """
        params = self.get_parameters(grad_only=False)
        nn.save_parameters(path, params=params, extension=extension)

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
                self.submodules[module_name].set_parameter(
                    subkey, param, raise_if_missing=raise_if_missing)
            elif raise_if_missing:
                raise ValueError(
                    'A child module {} cannot be found in {}. '
                    'This error is raised because `raise_if_missing` is specified '
                    'as True. Please turn off if you allow it.'.format(module_name, self))
            return

        # Set parameters
        with nn.parameter_scope('', self.parameter_scope):
            nn.parameter.set_parameter(key, param)

    def set_parameters(self, params, raise_if_missing=False):
        for key, param in params.items():
            self.set_parameter(key, param, raise_if_missing=raise_if_missing)

    def update_parameter(self):
        params = self.get_parameters()
        self.set_parameters(params)

    def zero_grad(self):
        '''
        Clear the gradient of the parameters in this module to 0.
        '''
        for param in self.get_parameters().values():
            param.grad.zero()

    def __len__(self):
        return len(self.submodules)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.submodules[name] = value
            value.__dict__['_name'] = name
            value.__dict__['_parent'] = self
            return
        self.__dict__[name] = value
        if name in self._recursive_attrs:
            for module in self.submodules.values():
                setattr(module, name, value)

    def __getattr__(self, name):
        if name in self.submodules:
            return self.submodules[name]
        attr = super().__getattr__(name)
        return attr

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
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        """User needs implement this function to construct their neural network.
        In the implementation, user may instantiate existing predefined Modules
        as its members, then use it. For example:

        .. code-block:: python

            class AModule(nn.Module):
               def __init__(...):
                  ...
                  self.cnb = ConvBN(128) # A submodule is instantiated here.

               def call(...):
                  h = self.cnb(x) # Using beforehand instantiated submodule.

        or directly use parametric functions or functions:

        .. code-block:: python

            class AModule(nn.Module):
                ...
                def call(...):
                    ...
                    h = PF.convolution(x, self.outmaps, ...)
                    return h

        Note:
              The following usage is currently not supported, it might be supported in future version:

              .. code-block:: python

                    class AModule(nn.Module):
                       def __init__(...):
                          ...
                          self.cnb = [ConvBN(k) for k in [8, 16, 32]] # using an array to hold module instances.
                          self.cnb = {f'name_{k}': ConvBN(k) for k in [8, 16, 32]} # using a dict to hold module instances.


        Note:
             The following method to temporarily instantiate a module is also not allowed:

             .. code-block:: python

                    class AModule(nn.Module):
                       def call(...):
                          ...
                          cnb = ConvBN(k) # Instantiate a temporary instance of Module is not allowed
                          y = cnb(x)
                          return y

             Because when leave this scope, the parameters registered to `cnb` module will be released, which cause
             unexpected result.

        """
        raise NotImplementedError('call(*av, **kw) must be implemented.')
