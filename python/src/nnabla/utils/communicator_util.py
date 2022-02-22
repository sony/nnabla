# Copyright 2018,2019,2020,2021 Sony Corporation.
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

import os
import nnabla.communicators as C
from nnabla.logger import logger

_current_communicator = None


def current_communicator():
    global _current_communicator
    return _current_communicator


def create_communicator(ignore_error=False, extension_module='cudnn', type_config='float'):
    global _current_communicator

    if os.environ.get('OMPI_COMM_WORLD_SIZE') is not None:
        from nnabla.ext_utils import get_extension_context
        context = get_extension_context(
            extension_module, type_config=type_config)
        try:
            logger.log(
                99, 'Create communicator with contexts {}'.format(context))
            _current_communicator = C.MultiProcessCommunicator(context)
            _current_communicator.init()
            context.device_id = str(_current_communicator.rank %
                                    _current_communicator.size)
            if _current_communicator.size == 1:
                _current_communicator = None
        except:
            if not ignore_error:
                raise
            logger.warning("Failed to initialize nnabla.communicators.")
            _current_communicator = None
    else:
        _current_communicator = None

    return _current_communicator


def single_or_rankzero():
    return not _current_communicator or _current_communicator.rank == 0
