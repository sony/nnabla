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


def check_arch_or_die(arch):
    # See available archs
    import sys
    from models.registry import get_available_archs
    archs = get_available_archs()
    if arch in archs:
        return
    print('Available architectures (spcify with -a option):')
    for an in archs:
        print('*', an)
    sys.exit(1)


def lower_str(value):
    if not isinstance(value, str):
        value = str(value)
    return value.lower()


def parse_tuple(x):
    return tuple(map(int, x.split(',')))


def add_runtime_args(parser):
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration.')
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension module. 'cudnn' is highly.recommended.")


def add_arch_args(parser):
    parser.add_argument('--arch', '-a', type=lower_str,
                        default='', help='Architecture type. See available choices for architecture by passing null string "".')


def get_train_args(
        max_epochs=90, learning_rate=1e-1, batch_size=256,
        weight_decay=1e-4, train_dir='./', train_list="train_label",
        val_dir='./', val_list="val_label", dali_num_threads=4):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os
    parser = argparse.ArgumentParser(
        description='''ImageNet classification example.
        ''')
    add_runtime_args(parser)
    add_arch_args(parser)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate,
                        help='Learning rate used when --batch-size=256. For other batch sizes, the learning rate is linearly scaled.')

    parser.add_argument("--learning-rate-decay-at", "-D",
                        default=(30, 60, 80), type=parse_tuple,
                        help='Step learning rate decay multiplied by 0.1 is performed at epochs specified, e.g. `-D 30,60,80`.')
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=None,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-epochs", "-e", type=int, default=max_epochs,
                        help='Max epochs of training.')
    parser.add_argument("--val-interval", "-v", type=int, default=10,
                        help='Evaluation with validation dataset is performed at every interval epochs specified.')
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay,
                        help='Weight decay factor of SGD update.')
    parser.add_argument("--warmup-epochs",
                        type=int, default=5,
                        help='Warmup learning rate during a specified number of epochs.')
    parser.add_argument("--model-save-interval", "-s", type=int, default=10,
                        help='The epoch interval of saving model parameters.')
    parser.add_argument("--model-load-path", type=str, default=None,
                        help='Path to the model parameters to be loaded.')
    parser.add_argument("--channel-last", action='store_true',
                        help='Use a model with NHWC layout.')
    parser.add_argument("--train-dir", '-T', type=str, default=train_dir,
                        help='Directory containing training data.')
    parser.add_argument("--train-list", type=str, default=train_list,
                        help='Training file list.')
    parser.add_argument("--val-dir", '-V', type=str, default=val_dir,
                        help='Directory containing validation data.')
    parser.add_argument("--val-list", type=str, default=val_list,
                        help='Validation file list.')
    parser.add_argument("--dali-num-threads", type=int, default=dali_num_threads,
                        help="DALI's number of CPU threads.")
    parser.add_argument('--dali-prefetch-queue', type=int,
                        default=2, help="DALI prefetch queue depth")
    parser.add_argument('--dali-nvjpeg-memory-padding-mb', type=int, default=64,
                        help="Memory padding value for nvJPEG (in MB)")

    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Ratio of label smoothing loss.')
    parser.add_argument('--loss-scaling', type=float, default=256,
                        help='Loss scaling value. Only used in half precision (mixed precision) training.')

    args = parser.parse_args()

    # Check arch is available
    check_arch_or_die(args.arch)

    if args.monitor_path is None:
        import datetime
        args.monitor_path = 'tmp.monitor.' + \
            datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # to bytes
    args.dali_nvjpeg_memory_padding = args.dali_nvjpeg_memory_padding_mb * \
        (1 << 20)

    # Learning rate is linearity scaled by batch size.
    args.learning_rate *= args.batch_size / 256.0

    return args
