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

import argparse

from utils import read_yaml


def get_config():
    """
    Get command line arguments.
    Arguments set the default values of command line arguments.
    """

    parser = argparse.ArgumentParser(description='TecoGAN')

    parser.add_argument('--cfg', default="./config.yaml")

    args, _ = parser.parse_known_args()

    conf = read_yaml(args.cfg)

    parser.add_argument('--input_video_dir', default=conf.data.input_video_dir,
                        help='The directory of the video input data, for training')
    parser.add_argument('--output_dir', default=conf.data.output_dir,
                        help='The output directory of the checkpoint')
    parser.add_argument('--num_resblock', type=int, default=conf.train.num_resblock,
                        help='How many residual blocks are there in the generator')
    parser.add_argument('--max_iter', type=int, default=conf.train.max_iter,
                        help='max iteration for training')
    parser.add_argument('--pre_trained_frvsr_weights', type=str, default=conf.train.pre_trained_frvsr_weights,
                        help='the weight of frvsr generator will be loaded as an initial point')
    parser.add_argument('--vgg_pre_trained_weights', type=str,
                        default=conf.train.vgg_pre_trained_weights,
                        help='path to pre-trained weights for the vgg19')
    parser.add_argument('--tecogan', type=bool,
                        default=conf.train.tecogan,
                        help='True for Tecogan training False for FRVSR training')
    parser.add_argument('--checkpoint', type=str,
                        default=conf.train.checkpoint, help='path to checkpoint file')
    args = parser.parse_args()

    # refine config
    conf.data.input_video_dir = args.input_video_dir
    conf.train.max_iter = conf.train.max_iter
    conf.data.output_dir = args.output_dir
    conf.train.num_resblock = args.num_resblock
    conf.train.pre_trained_frvsr_weights = args.pre_trained_frvsr_weights
    conf.train.vgg_pre_trained_weights = args.vgg_pre_trained_weights
    conf.train.tecogan = args.tecogan
    conf.train.checkpoint = args.checkpoint

    return conf
