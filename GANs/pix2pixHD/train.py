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

import nnabla as nn

from trainer import Trainer
from args import get_training_args as get_args
from utils import CommunicatorWrapper, get_cityscape_datalist as get_datalist

if __name__ == '__main__':
    args = get_args()

    # set context
    from nnabla.ext_utils import get_extension_context

    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)

    # init communicator
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    # disable outputs from logger except rank==0
    if comm.rank > 0:
        from nnabla import logger
        import logging

        logger.setLevel(logging.ERROR)

    # setup dataset
    data_list = get_datalist(args, save_file=comm.rank == 0)

    # define trainer
    trainer = Trainer(batch_size=args.batch_size,
                      base_image_shape=(512, 1024),
                      data_list=data_list,
                      max_epoch=args.epoch,
                      learning_rate=args.base_lr,
                      comm=comm,
                      fix_global_epoch=args.fix_global_epoch,
                      d_n_scales=args.d_n_scales,
                      g_n_scales=args.g_n_scales,
                      n_label_ids=args.n_label_ids,
                      use_encoder=not args.without_flip,
                      load_path=args.load_path,
                      save_path=args.save_path,
                      is_data_flip=not args.without_flip)

    trainer.train()
