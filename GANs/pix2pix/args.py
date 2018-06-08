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


def get_args():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Pix2pix NNabla Code')
    parser.add_argument('--traindir',
                        type=str,
                        default='./datasets/facades/train',
                        help='path to images root directory')
    parser.add_argument('--valdir',
                        type=str,
                        default='./datasets/facades/val',
                        help='path to images root directory')
    parser.add_argument('--testdir',
                        type=str,
                        default='./datasets/facades/test',
                        help='path to images root directory')
    parser.add_argument('--model',
                        type=str,
                        default=None,
                        help='path to saved model')
    parser.add_argument('--logdir', '-l',
                        type=str,
                        default='tmp.monitor',
                        help='path to monitor files directory')
    parser.add_argument('--epoch', '-e',
                        type=int,
                        default=800,
                        help='max epoch for training (default: 200)')
    parser.add_argument('--batchsize', '-b',
                        type=int,
                        default=1,
                        help='batchsize for training and test (default: 1)')
    parser.add_argument('--lrate', '-r',
                        type=float,
                        default=2e-4,
                        help='learning rate for training (default: 2e-4)')
    parser.add_argument('--beta1',
                        type=float,
                        default=0.5,
                        help='learning rate for training (default: 0.5)')
    parser.add_argument('--weight-l1',
                        type=float,
                        default=100,
                        help='weight on L1 loss of generator (default: 100)')
    parser.add_argument('--patch-gan',
                        action='store_true',
                        default=False,
                        help='PatchGAN discriminator flag')
    parser.add_argument('--monitor-interval', '-m',
                        type=int,
                        default=500,
                        help='Interval of monitoring (default: 500)')
    parser.add_argument('--context', '-c',
                        choices={'cpu', 'cudnn'},
                        default='cpu',
                        help='device context (default: cpu)')
    parser.add_argument('--device-id', '-d',
                        type=int,
                        default=0,
                        help='device id (default: 0)')
    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        help='Training mode flag')
    parser.add_argument('--generate',
                        action='store_true',
                        default=False,
                        help='Generating mode flag')

    args = parser.parse_args()
    return args
