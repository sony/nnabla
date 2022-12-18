# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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


def extension_context(extension_name='cpu', **kw):
    """Get the context of the specified extension.

    All extension's module must provide `context(**kw)` function.

    Args:
        extension_name (str) : Module path relative to `nnabla_ext`.
        kw (dict) : Additional keyword arguments for context function in a extension module.

    Returns:
        :class:`nnabla.Context`: The current extension context.

    Note:
        Deprecated. Use :function:`nnabla.ext_utils.get_extension_context` instead.

    Example:

        .. code-block:: python

            ctx = extension_context('cuda.cudnn', device_id=0)
            nn.set_default_context(ctx)

    """
    from nnabla import logger
    logger.warning(
        'Deprecated API. Use `nnabla.ext_util.get_extension_context(ext_name, **kw)`.')
    from nnabla.ext_utils import get_extension_context
    return get_extension_context(extension_name, **kw)
