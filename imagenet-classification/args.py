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


def get_args(monitor_path='tmp.monitor.imagenet', max_iter=500000, model_save_path=None, learning_rate=1e-1, batch_size=8, weight_decay=1e-4, accum_grad=32, tiny_mode=False, train_cachefile_dir=None, val_cachefile_dir=None, warmup_epoch=5):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    import os
    if model_save_path is None:
        model_save_path = monitor_path
    parser = argparse.ArgumentParser(
        description='''(Tiny) ImageNet classification example.
        ''')
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--accum-grad", "-a", type=int, default=accum_grad,
                        help='Parameters are updated by the gradient accumulated by multiple mini-batches.')
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)

    def parse_tuple(x):
        return tuple(map(int, x.split(',')))
    parser.add_argument("--learning-rate-decay-at", "-D",
                        default=(150000, 300000, 450000), type=parse_tuple,
                        help='Execution point of learning rate decay with format(x1,x2,,xn). Learning rate will multiplied by 0.1 at the iteration specified.')
    parser.add_argument("--monitor-path", "-m",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help='Max iteration of training.')
    parser.add_argument("--val-interval", "-v", type=int, default=100,
                        help='Validation interval.')
    parser.add_argument("--val-iter", "-j", type=int, default=10,
                        help='Each validation runs `val_iter mini-batch iteration.')
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay,
                        help='Weight decay factor of SGD update.')
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type configuration.')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000,
                        help='The interval of saving model parameters.')
    parser.add_argument("--model-save-path", "-o",
                        type=str, default=model_save_path,
                        help='Path the model parameters saved.')
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension module. 'cudnn' is highly.recommended.")
    parser.add_argument("--num-layers", "-L", type=int,
                        choices=[18, 34, 50, 101, 152], default=34,
                        help='Number of layers of ResNet.')
    parser.add_argument("--shortcut-type", "-S", type=str,
                        choices=['b', 'c', ''], default='b',
                        help='Skip connection type. See `resnet_imagenet()` in model_resent.py for description.')
    parser.add_argument("--tiny-mode", "-M", type=bool, default=tiny_mode,
                        help='The dataset is tiny imagenet.')
    parser.add_argument("--train-cachefile-dir", "-T", type=str, default=train_cachefile_dir,
                        help='Training cache file dir. Create to use create_cache_file.py')
    parser.add_argument("--val-cachefile-dir", "-V", type=str, default=val_cachefile_dir,
                        help='Validation cache file dir. Create to use create_cache_file.py')

    args = parser.parse_args()
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    return args
