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
    parser = argparse.ArgumentParser(description='ESRGAN')
    parser.add_argument('--cfg', default="./config.yaml")
    args, subargs = parser.parse_known_args()
    conf = read_yaml(args.cfg)
    parser.add_argument('--gt_train', default=conf.DIV2K.gt_train,
                        help='train ground truth (HQ) image path')
    parser.add_argument('--lq_train', default=conf.DIV2K.lq_train,
                        help='train low quality (LQ) image path')
    parser.add_argument('--gt_val', default=conf.SET14.gt_val,
                        help='val ground truth (HQ) image path')
    parser.add_argument('--lq_val', default=conf.SET14.lq_val,
                        help='val generated images')
    parser.add_argument('--save_results', default=conf.val.save_results,
                        help='path for saving validation results.')
    parser.add_argument('--esrgan', default=conf.model.esrgan,
                        help='True to train esrgan')
    parser.add_argument('--n_epochs', type=int, default=conf.train.n_epochs,
                        help='no. of epochs 497 for psnr oriented rrdb and 199 for esrgan')
    parser.add_argument('--lr_g', type=int, default=conf.hyperparameters.lr_g,
                        help='initial generator learning rate 2e-4 for PSNR, 1e-4 for esrgan')
    parser.add_argument('--savemodel', default=conf.train.savemodel,
                        help='path to save the trained weights')
    parser.add_argument('--vgg_pre_trained_weights', type=str,
                        default=conf.train.vgg_pre_trained_weights,
                        help="Path to VGG19 weights")
    parser.add_argument('--gen_pretrained', type=str,
                        default=conf.train.gen_pretrained,
                        help="path to psnr rrdb pretrained file")
    args = parser.parse_args()

    # refine config
    conf.DIV2K.lq_train = args.lq_train
    conf.DIV2K.gt_train = args.gt_train
    conf.SET14.gt_val = args.gt_val
    conf.SET14.lq_val = args.lq_val
    conf.train.n_epochs = args.n_epochs
    conf.hyperparameters.lr_g = args.lr_g
    conf.train.savemodel = args.savemodel
    conf.model.esrgan = args.esrgan
    conf.train.vgg_pre_trained_weights = args.vgg_pre_trained_weights
    conf.train.psnr_rrdb_pretrained = args.gen_pretrained
    conf.val.save_results = args.save_results
    return conf


if __name__ == "__main__":
    conf = get_config()
