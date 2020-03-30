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
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of categories of classification.')


def add_train_dataset_args(parser, train_dir='./', train_list="train_label"):
    parser.add_argument("--train-dir", '-T', type=str, default=train_dir,
                        help='Directory containing training data.')
    parser.add_argument("--train-list", type=str, default=train_list,
                        help='Training file list.')


def add_val_dataset_args(parser, val_dir='./', val_list="val_label"):
    parser.add_argument("--val-dir", '-V', type=str, default=val_dir,
                        help='Directory containing validation data.')
    parser.add_argument("--val-list", type=str, default=val_list,
                        help='Validation file list.')


def add_dataset_args(parser):
    add_train_dataset_args(parser)
    add_val_dataset_args(parser)


def add_training_args(parser):
    parser.add_argument("--batch-size", "-b", type=int, default=128,
                        help='Batch size per worker. The default is 128.')
    parser.add_argument("--epochs", "-e", type=int, default=None,
                        help='Number of epochs for training. It overwrites the config described by `--train-config`.')
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=None,
                        help='Path monitoring logs saved.')
    parser.add_argument("--val-interval", "-v", type=int, default=10,
                        help='Evaluation with validation dataset is performed at every interval epochs specified.')
    parser.add_argument("--model-save-interval", "-s", type=int, default=10,
                        help='The epoch interval of saving model parameters.')
    parser.add_argument("--model-load-path", type=str, default=None,
                        help='Path to the model parameters to be loaded.')
    parser.add_argument('--train-config', '-C', type=str, default='cfg/train_default.yaml',
                        help='A config file which describes optimization configuration such as default batch size, solver, number of epochs, and learning rate scheduling.')


def mb_to_b(mb):
    return int(mb) * (1 << 20)


def add_dali_args(parser):
    parser.add_argument("--dali-num-threads", type=int, default=4,
                        help="DALI's number of CPU threads.")
    parser.add_argument('--dali-prefetch-queue', type=int,
                        default=2, help="DALI prefetch queue depth")
    parser.add_argument('--dali-nvjpeg-memory-padding-mb', type=mb_to_b, default=64,
                        dest='dali_nvjpeg_memory_padding',
                        help="Memory padding value for nvJPEG (in MB)")


def get_train_args():
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
    parser.add_argument("--channel-last", action='store_true',
                        help='Use a model with NHWC layout.')
    add_training_args(parser)
    add_dataset_args(parser)
    add_dali_args(parser)

    args = parser.parse_args()

    # Check arch is available
    check_arch_or_die(args.arch)

    if args.monitor_path is None:
        import datetime
        args.monitor_path = 'tmp.monitor.' + \
            datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    from utils import read_yaml
    train_config = read_yaml(args.train_config)
    if args.epochs is not None:
        train_config.epochs = args.epochs
    return args, train_config
