# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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


class Module(object):
    """Module mix-in for the parametric function classes.
    """

    def __init__(self):
        pass

    def get_parameters(self, grad_only=True):
        """Get parameters.
        Args:
            grad_only (bool, optional): Return parameters with `need_grad` option as `True`. 
            If you set this option as `False`, All parameters are returned. Default is `True`.
        Returns:
            dict: The dictionary of parameter name (`str`) to Variable (:obj:`~nnabla.Variable`).
        """
        params = OrderedDict()

        for v in self.get_modules():
            if not isinstance(v, tuple):
                continue
            prefix, module = v
            for k, v in module.__dict__.items():
                if not isinstance(v, nn.Variable):
                    continue
                pname = k
                name = "{}/{}".format(prefix, pname)
                if grad_only and v.need_grad == False:
                    continue
                params[name] = v
        return params

    def get_modules(self, memo=None, prefix=""):
        """Get modules.

        This function is internally used as the helper method for other methods.

        Args: 
            memo (set, optional): Module set in order to memorize to visit.
            prefix (str, optional): Prefix to a specific parameter name.

        Yields:
            `Module`: The module class.
        """
        if memo is None:
            memo = set()

        if self not in memo:
            memo.add(self)
            yield prefix, self
            for k, v in self.__dict__.items():
                if not isinstance(v, Module):
                    continue
                name, module = k, v
                submodule_prefix = "{}/{}".format(prefix,
                                                  name) if prefix != "" else name
                for m in module.get_modules(memo, submodule_prefix):
                    yield m

    def save_parameters(self, path, grad_only=False):
        """Save all parameters into a file with the specified format.

        Currently hdf5 and protobuf formats are supported.

        Args:
            path : path or file object
            grad_only (bool, optional): Return parameters with `need_grad` option as `True`. 
        """
        params = self.get_parameters(grad_only=grad_only)
        nn.save_parameters(path, params)

    def load_parameters(self, path):
        """Load parameters from a file with the specified format.

        Args:
            path : path or file object
        """
        nn.load_parameters(path)
        for v in self.get_modules():
            if not isinstance(v, tuple):
                continue
            prefix, module = v
            for k, v in module.__dict__.items():
                if not isinstance(v, nn.Variable):
                    continue
                pname = k
                name = "{}/{}".format(prefix, pname)
                # Substitute
                param0 = v
                param1 = nn.parameter.pop_parameter(name)
                if param0 is None:
                    raise ValueError(
                        "Model does not have {} parameter.".format(name))
                param0.d = param1.d.copy()
                nn.logger.info("`{}` loaded.)".format(name))
