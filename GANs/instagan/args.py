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


def get_args():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default="datasets/jeans2skirt_ccp",
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc).')
    # image size.
    parser.add_argument('--loadSizeW', type=int, default=220,
                        help='Loaded image width from dataset.')
    parser.add_argument('--loadSizeH', type=int, default=330,
                        help='Loaded image height from dataset.')
    parser.add_argument('--fineSizeW', type=int, default=200,
                        help='Image width after crop. Must be less than loadSizeW')
    parser.add_argument('--fineSizeH', type=int, default=300,
                        help='Image height after crop. Must be less than loadSizeH.')

    # network config.
    parser.add_argument('--input-nc', type=int, default=3,
                        help='number of input image channels.')
    parser.add_argument('--output-nc', type=int, default=3,
                        help='number of output image channels.')
    parser.add_argument('--ngf', type=int, default=64,
                        help='number of generator filters in first conv layer.')
    parser.add_argument('--ndf', type=int, default=64,
                        help='number of discriminator filters in first conv layer.')
    parser.add_argument('--n-layers-D', type=int, default=3,
                        help='number of layers for discriminator.')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization.')
    parser.add_argument('--no-flip', action='store_true',
                        help='if specified, do not flip the images for data augmentation.')

    # training config.
    parser.add_argument('--context', '-c', type=str, default="cudnn",
                        help="Extension modules. ex) 'cpu', 'cudnn'.")
    parser.add_argument("--type-config", "-t", type=str,
                        default='float', help='Type configuration (float or half).')
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--learning-rate_G", type=float, default=0.0002)
    parser.add_argument("--learning-rate_D", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--monitor-path", "-m", type=str,
                        default="tmp.monitor", help='Path monitoring logs saved.')
    parser.add_argument("--max-epoch", "-e", type=int, default=600,
                        help='Max epoch of training. Epoch is determined by the max of the number of images for two domains.')
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--model-save-path", "-o", type=str,
                        default=None, help='Path where model parameters are saved.')
    parser.add_argument("--lambda-cyc", type=float, default=10.)
    parser.add_argument("--lambda-idt", type=float, default=10.)
    parser.add_argument("--lambda-ctx", type=float, default=10.)
    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--save-image-interval', type=int, default=1000)
    parser.add_argument("--lr-decay-start-epoch", type=int, default=400)

    args = parser.parse_args()

    if args.model_save_path is None:
        args.model_save_path = args.monitor_path

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)

    return args
