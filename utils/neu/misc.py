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


def init_nnabla(ctx_config):
    import nnabla as nn
    from nnabla.ext_utils import get_extension_context
    from .comm import CommunicatorWrapper

    # set context
    ctx = get_extension_context(**ctx_config)

    # init communicator
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    # disable outputs from logger except rank==0
    if comm.rank > 0:
        from nnabla import logger
        import logging

        logger.setLevel(logging.ERROR)

    return comm


class AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("No such attribute `{}`".format(key))

        if isinstance(self[key], dict):
            self[key] = AttrDict(self[key])

        return self[key]

    def dump_to_stdout(self):
        print("================================configs================================")
        for k, v in self.items():
            print("{}: {}".format(k, v))

        print("=======================================================================")


def makedirs(dirpath):
    if os.path.exists(dirpath):
        if os.path.isdir(dirpath):
            return
        else:
            raise ValueError(
                "{} already exists as a file not a directory.".format(dirpath))

    os.makedirs(dirpath)


def get_current_time():
    from datetime import datetime

    return datetime.now().strftime('%m%d_%H%M%S')
