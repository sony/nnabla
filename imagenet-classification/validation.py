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
import nnabla.functions as F
from nnabla import logger

import numpy as np

from utils import (
    ceil_to_multiple,
    CommunicatorWrapper,
)
from data import get_val_data_iterator
from infer import load_parameters_and_config
from train import get_model


def get_args():
    import argparse
    import args as A
    parser = argparse.ArgumentParser(
        description='Inference.')
    A.add_runtime_args(parser)
    A.add_arch_args(parser)
    A.add_val_dataset_args(parser)
    A.add_dali_args(parser)
    parser.add_argument('--batch-size', '-b', default=100,
                        type=int, help='Batch size per GPU.')
    parser.add_argument('--monitor-path', '-M', default='tmp.val',
                        help='A directory the results are produced.')
    parser.add_argument("weights", help='Path to a trained parameter h5 file.')
    parser.add_argument(
        '--labels', help='Path to a label name file which contain label names as csv compatible with `label_words.csv`.', default='./label_words.csv')
    parser.add_argument(
        '--norm-config', '-n', type=A.lower_str, default='default',
        help='Specify how to normalize an image as preprocessing.')
    parser.add_argument(
        '--disable-dataset-size-error', '-E', action='store_false', default=True, dest='raise_dataset_size',
        help='If not set, raising an error when the dataset size is divisible by (batch_size * num workers)')
    args = parser.parse_args()

    # See available archs
    A.check_arch_or_die(args.arch)
    return args


def main():
    args = get_args()

    # Setup
    from nnabla.ext_utils import get_extension_context
    if args.context is None:
        print('Computation backend is not specified. Using the default "cudnn".')
        extension_module = "cudnn"
    else:
        extension_module = args.context
    ctx = get_extension_context(
        extension_module, device_id=args.device_id, type_config=args.type_config)
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(comm.ctx)

    if args.raise_dataset_size:
        imagenet_val_size = 50000
        if imagenet_val_size % (comm.n_procs * args.batch_size) != 0:
            raise ValueError(f'The batchsize and number of workers must be set so that {imagenet_val_size} can be divisible by (batch_size * num_workers).')

    # Load parameters
    channel_last, channels = load_parameters_and_config(args.weights)
    logger.info('Parameter configuration is inferred as:')
    logger.info(f'* channel_last={channel_last}')
    logger.info(f'* channels={channels}')
    args.channel_last = channel_last

    # Build a validation entwork
    from models import build_network
    num_classes = args.num_classes
    # Network for validation
    v_model = get_model(args, num_classes,
                        test=True, channel_last=channel_last,
                        channels=channels)

    vdata = get_val_data_iterator(args, comm, channels, args.norm_config)

    from nnabla_ext.cuda import StreamEventHandler
    stream_event_handler = StreamEventHandler(int(comm.ctx.device_id))

    # Monitors
    import nnabla.monitor as M
    import os
    monitor = None
    if comm.rank == 0:
        if not os.path.isdir(args.monitor_path):
            os.makedirs(args.monitor_path)
        monitor = M.Monitor(args.monitor_path)

    from utils import EpochValidator
    EpochValidator(v_model, vdata, comm, monitor, stream_event_handler).run(0)


if __name__ == '__main__':
    main()
