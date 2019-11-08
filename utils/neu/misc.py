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


def init_nnabla(conf):
    import nnabla as nn
    from nnabla.ext_utils import get_extension_context
    from .comm import CommunicatorWrapper

    # set context
    ctx = get_extension_context(ext_name=conf.ext_name, device_id=conf.device_id, type_config=conf.type_config)

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
    # special internal variable used for error message.
    _parent = []

    def __setattr__(self, key, value):
        if key == "_parent":
            self.__dict__["_parent"] = value

        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(
                "dict (AttrDict) has no chain of attributes '{}'".format(".".join(self._parent + [key])))

        if isinstance(self[key], dict):
            self[key] = AttrDict(self[key])
            self[key]._parent = self._parent + [key]

        return self[key]

    def __str__(self):
        if "_parent" in self:
            del self["_parent"]

        return super(AttrDict, self).__str__()

    def __repr__(self):
        if "_parent" in self:
            del self["_parent"]

        return super(AttrDict, self).__repr__()

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


def get_iteration_per_epoch(dataset_size, batch_size, round="ceil"):
    """
    Calculate a number of iterations to see whole images in dataset (= 1 epoch).

    Args:
     dataset_size (int): A number of images in dataset
     batch_size (int): A number of batch_size.
     round (str): Round method. One of ["ceil", "floor"].

    return: int
    """
    import numpy as np

    round_func = {"ceil": np.ceil, "floor": np.floor}
    if round not in round_func:
        raise ValueError("Unknown rounding method {}. must be one of {}.".format(round,
                                                                                   list(round_func.keys())))

    ipe = float(dataset_size) / batch_size

    return int(round_func[round](ipe))
