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


def get_args(monitor_path='tmp.monitor', max_epoch=600):

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c-dim', type=int, default=5,
                        help='dimension of domain labels')
    parser.add_argument('--num-downsample', type=int, default=2,
                        help='number of downsample')
    parser.add_argument('--num-upsample', type=int, default=2,
                        help='number of upsample')
    parser.add_argument('--celeba-crop-size', type=int,
                        default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--image-size', type=int,
                        default=128, help='image resolution')
    parser.add_argument('--g-conv-dim', type=int, default=64,
                        help='number of conv filters in the first layer of G')
    parser.add_argument('--d-conv-dim', type=int, default=64,
                        help='number of conv filters in the first layer of D')
    parser.add_argument('--g-repeat-num', type=int, default=6,
                        help='number of residual blocks in G')
    parser.add_argument('--d-repeat-num', type=int, default=6,
                        help='number of strided conv layers in D')

    # Training configuration.
    parser.add_argument('--context', '-c', type=str,
                        default='cudnn', help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument('--batch-size', type=int,
                        default=8, help='mini-batch size')
    parser.add_argument('--max-epoch', type=int, default=20,
                        help='number of max epoch for training D')
    parser.add_argument('--g-lr', type=float, default=0.0001,
                        help='learning rate for G')
    parser.add_argument('--d-lr', type=float, default=0.0001,
                        help='learning rate for D')
    parser.add_argument('--n-critic', type=int, default=5,
                        help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for Adam optimizer')
    parser.add_argument('--selected-attrs', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--lambda-cls', type=float, default=1,
                        help='weight for domain classification loss')
    parser.add_argument('--lambda-rec', type=float, default=10,
                        help='weight for reconstruction loss')
    parser.add_argument('--lambda-gp', type=float, default=10,
                        help='weight for gradient penalty')
    parser.add_argument('--test-during-training', action='store_true',
                        help="if True, save translated images from test dataset")

    # Dataset Stuff.
    parser.add_argument('--num-data', type=int, default=None,
                        help='Number of data used for training. Since CelebA datasetcontains so many images, you may need to limit its size.')
    parser.add_argument('--num-test', type=int, default=30,
                        help='Number of testing of image translation.')
    parser.add_argument('--celeba-image-dir', type=str,
                        default='data/celeba/images')
    parser.add_argument('--attr-path', type=str,
                        default='data/celeba/list_attr_celeba.txt')

    # Step size.
    parser.add_argument('--log-step', type=int, default=10)
    parser.add_argument('--sample-step', type=int, default=1000,
                        help="Note that this needs to be mutiple of numbers set by n_critic.")
    parser.add_argument('--lr-update-step', type=int, default=1000)

    # Monitor Path.
    parser.add_argument('--monitor-path', type=str, default=monitor_path)
    parser.add_argument('--model-save-path', type=str, default=monitor_path)

    args = parser.parse_args()
    if not os.path.isdir(args.monitor_path):
        os.makedirs(args.monitor_path)

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    return args
