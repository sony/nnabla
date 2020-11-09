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

import os
from .yaml_wrapper import write_yaml


def save_args(args, config=None):
    from nnabla import logger
    if not os.path.exists(args.monitor_path):
        os.makedirs(args.monitor_path)

    path = os.path.join(args.monitor_path, "Arguments.txt")
    logger.info("Arguments are saved to {}.".format(path))
    with open(path, "w") as fp:
        for k, v in sorted(vars(args).items()):
            logger.info("{}={}".format(k, v))
            fp.write("{}={}\n".format(k, v))

            if config is not None and isinstance(v, str) and v.endswith(".yaml"):
                path = os.path.join(args.monitor_path, v)
                write_yaml(path, config)
