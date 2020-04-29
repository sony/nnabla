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
import sys
import os
from utils import *


def get_config():

    parser = argparse.ArgumentParser(description='TECOGAN')
    parser.add_argument('--cfg', default="./config.yaml")
    args, subargs = parser.parse_known_args()
    conf = read_yaml(args.cfg)
    parser.add_argument('--input_video_dir', default=conf.data.input_video_dir,
                        help='The directory of the video input data, for training')
    parser.add_argument('--mode', default='conf.train.mode',
                        help='train, or inference')
    parser.add_argument('--output_dir', default=conf.data.output_dir,
                        help='The output directory of the checkpoint')
    parser.add_argument('--num_resblock', type=int, default=conf.train.num_resblock,
                        help='How many residual blocks are there in the generator')
    parser.add_argument('--pre_trained_model', type=str, default=conf.train.pre_trained_model,
                        help='the weight of generator will be loaded as an initial point')
    parser.add_argument('--vgg_pre_trained_weights', type=str, default=conf.train.vgg_pre_trained_weights,
                        help='path to pre-trained weights for the vgg19')
    parser.add_argument('--tecogan', type=bool,
                        default=conf.train.tecogan, help='True for Tecogan training False for FRVSR training')
    parser.add_argument('--input_dir_LR', default=conf.data.input_dir_LR,
                        help='The directory of the input resolution input data, for inference mode')
    parser.add_argument('--input_dir_HR', default=conf.data.input_dir_HR,
                        help='The directory of the input resolution input data, for inference mode')

    args = parser.parse_args()


    # refine config
    conf.data.input_video_dir = args.input_video_dir
    conf.data.mode = args.mode
    conf.data.output_dir = args.output_dir
    conf.train.num_resblock = args.num_resblock
    conf.train.pre_trained_model = args.pre_trained_model
    conf.train.vgg_pre_trained_weights = args.vgg_pre_trained_weights
    conf.train.tecogan = args.tecogan
    conf.data.input_dir_LR = args.input_dir_LR
    conf.data.input_dir_HR = args.input_dir_HR
    return conf


if __name__ == "__main__":
    conf = get_config()
