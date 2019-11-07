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

import argparse
import os

from trainer import Trainer
from utils import *


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.yaml")
    args, subargs = parser.parse_known_args()

    conf = read_yaml(args.cfg)

    # nnabla execution context args
    parser.add_argument(
        "--device-id", default=conf.nnabla_context.device_id, type=int)
    parser.add_argument("--ext_name", default=conf.nnabla_context.ext_name)
    parser.add_argument(
        "--type-config", default=conf.nnabla_context.type_config)

    # training args
    parser.add_argument("--fix-global-epoch", default=conf.train.fix_global_epoch, type=int,
                        help="Number of epochs where the global generator's parameters are fixed.")
    parser.add_argument("--save-path", default=conf.train.save_path)
    parser.add_argument("--load-path", default=conf.train.load_path)

    # model args
    parser.add_argument("--d-n-scales", default=conf.model.d_n_scales, type=int,
                        help="Number of layers of discriminator pyramids")
    parser.add_argument("--g-n-scales", default=conf.model.g_n_scales, type=int,
                        help="A number of generator resolution stacks. If 1, only global generator is used.")

    args = parser.parse_args()

    # refine config
    conf.nnabla_context.update(
        {"device_id": args.device_id, "ext_name": args.ext_name, "type_config": args.type_config})
    conf.train.fix_global_epoch = args.fix_global_epoch
    conf.model.d_n_scales = args.d_n_scales
    conf.model.g_n_scales = args.g_n_scales

    conf.train.save_path = args.save_path

    return conf


if __name__ == '__main__':
    conf = get_config()

    comm = init_nnabla(conf.nnabla_context)

    # setup dataset
    if conf.train.data_set == "cityscapes":
        data_list = get_cityscape_datalist(
            conf.cityscapes, save_file=comm.rank == 0)
        conf.model.n_label_ids = conf.cityscapes.n_label_ids
    else:
        raise NotImplementedError(
            "Currently dataset {} is not supported.".format(conf.dataset))

    # dump conf to file
    if comm.rank == 0:
        write_yaml(os.path.join(conf.train.save_path, "config.yaml"), conf)

    # define trainer
    trainer = Trainer(conf.train, conf.model, comm, data_list)

    trainer.train()
