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


def get_args(save_path='tmp.monitor', max_iter=80000, learning_rate=1e-1, batch_size=128):
    """
    Get command line arguments.

    Arguments set the default values of command line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-l",
                        type=float, default=learning_rate)
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter)
    parser.add_argument("--val-interval", "-v", type=int, default=1000)
    parser.add_argument("--device-id", "-g", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--model-save-interval", "-s", type=int, default=5000)
    parser.add_argument("--save-path", "-p",
                        type=str, default=save_path)
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--net", "-n", type=str,
                        default='resnet18',
                        help="Neural network architecture type.")
    parser.add_argument("--solver", "-o", type=str,
                        default='Adam',
                        help="Solver for training DNN")

    # Mix-up
    parser.add_argument("--alpha", "-a", type=float, default=0.5)
    parser.add_argument("--mixtype", "-mt", type=str, default="mixup")

    return parser.parse_args()
