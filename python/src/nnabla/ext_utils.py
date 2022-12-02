# Copyright 2018,2019,2020,2021 Sony Corporation.
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

"""
Utilities for NNabla extensions.
"""


def import_extension_module(ext_name):
    """
    Import an extension module by name.

    The extension modules are installed under the `nnabla_ext` package as
    namespace packages. All extension modules provide a unified set of APIs.

    Args:
        ext_name(str): Extension name. e.g. 'cpu', 'cuda', 'cudnn' etc.

    Returns: module
        An Python module of a particular NNabla extension.

    Example:

        .. code-block:: python

            ext = import_extension_module('cudnn')
            available_devices = ext.get_devices()
            print(available_devices)
            ext.device_synchronize(available_devices[0])
            ext.clear_memory_cache()

    """
    import importlib
    try:
        return importlib.import_module('.' + ext_name, 'nnabla_ext')
    except ImportError as e:
        from nnabla import logger
        logger.error('Extension `{}` does not exist.'.format(ext_name))
        raise e


def list_extensions():
    """
    List up available extensions.

    Note:
        It may not work on some platforms/environments since it depends
        on the directory structure of the namespace packages.

    Returns: list of str
        Names of available extensions.

    """
    import nnabla_ext.cpu
    from os.path import dirname, join, realpath
    from os import listdir
    ext_dir = realpath((join(dirname(nnabla_ext.cpu.__file__), '..')))
    return listdir(ext_dir)


def get_extension_context(ext_name, **kw):
    """Get the context of the specified extension.

    All extension's module must provide `context(**kw)` function.

    Args:
        ext_name (str) : Module path relative to `nnabla_ext`.
        kw (dict) : Additional keyword arguments for context function in a extension module.

    Returns:
        :class:`nnabla.Context`: The current extension context.

    Example:

        .. code-block:: python

            ctx = get_extension_context('cudnn', device_id='0', type_config='half')
            nn.set_default_context(ctx)

    """
    if ext_name == 'cuda.cudnn':
        from nnabla import logger
        logger.warning(
            'Deprecated extension name "cuda.cudnn" passed. Use "cudnn" instead.')
        ext_name = 'cudnn'
    mod = import_extension_module(ext_name)
    return mod.context(**kw)
