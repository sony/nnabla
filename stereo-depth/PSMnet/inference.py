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
import random
import numpy as np
import scipy.stats as sp
import time
import os
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_iterator import data_iterator_simple
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from stackhourglass import psm_net
from datetime import datetime
import scipy.misc
import cv2
from nnabla.utils.image_utils import imsave, imread

parser = argparse.ArgumentParser(description='psm_net')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='max disparity')
parser.add_argument('--loadmodel', default='./psmnet_trained_param_293.h5',
                    help='load model')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--im_width_sf', type=int, default=960,
                    help='set image width')
parser.add_argument('--im_height_sf', type=int, default=512,
                    help='set  image height')
parser.add_argument('--im_width_kt', type=int, default=1232,
                    help='set image width')
parser.add_argument('--im_height_kt', type=int, default=368,
                    help='set  image height')
parser.add_argument('-g', '--gpus', type=str,
                    help='GPU IDs to be used', default="0")
parser.add_argument('-l', '--left_image', type=str,
                    help='path to left input image to model')
parser.add_argument('-r', '--right_image', type=str,
                    help='path to right input image to model')
parser.add_argument('--dataset', type=str,
                    help='select dataset from "SceneFlow" or "Kitti"', default="Kitti")
parser.add_argument('--nnp', type=str, default='psmnet_kitti.nnp')
parser.add_argument('--save-nnp', type=bool, default=False)
args = parser.parse_args()


def preprocess_kitti(image_left, image_right):
    w, h = image_left.shape[1], image_left.shape[0]
    image_l = image_left[h - args.im_heigh_kt:h, w-args.im_width_kt:w, :]
    image_r = image_right[h - args.im_height_kt:h, w-args.im_width_kt:w, :]
    image_l, image_r = preprocess_common(image_l, image_r)
    return image_l, image_r


def preprocess_sceneflow(image_left, image_right):
    image_l = image_left[:args.im_height_sf, :args.im_width_sf, :]
    image_r = image_right[:args.im_height_sf, :args.im_width_sf, :]
    image_l, image_r = preprocess_common(image_l, image_r)
    return image_l, image_r


def preprocess_common(image_left, image_right):
    mean_imagenet = np.asarray([0.485, 0.456, 0.406]).astype(
        np.float32).reshape(3, 1, 1)
    std_imagenet = np.asarray([0.229, 0.224, 0.225]).astype(
        np.float32).reshape(3, 1, 1)
    image_left, image_right = np.rollaxis(
        image_left, 2), np.rollaxis(image_right, 2)
    image_left = (image_left/255).astype(np.float32)
    image_right = (image_right/255).astype(np.float32)
    image_left -= mean_imagenet
    image_left /= std_imagenet
    image_right -= mean_imagenet
    image_right /= std_imagenet
    return image_left, image_right


def main():
    ctx = get_extension_context('cudnn', device_id=args.gpus)
    nn.set_default_context(ctx)
    image_left = imread(args.left_image)
    image_right = imread(args.right_image)

    if args.dataset == 'Kitti':
        var_left = nn.Variable((1, 3, args.im_height_kt, args.im_width_kt))
        var_right = nn.Variable((1, 3, args.im_height_kt, args.im_width_kt))
        img_left, img_right = preprocess_kitti(image_left, image_right)
    elif args.dataset == 'SceneFlow':
        var_left = nn.Variable((1, 3, args.im_height_sf, args.im_width_sf))
        var_right = nn.Variable((1, 3, args.im_height_sf, args.im_width_sf))
        img_left, img_right = preprocess_sceneflow(image_left, image_right)

    var_left.d, var_right.d = img_left, img_right
    if args.loadmodel is not None:
        # Loading CNN pretrained parameters.
        nn.load_parameters(args.loadmodel)
    pred_test = psm_net(var_left, var_right, args.maxdisp, False)
    pred_test.forward(clear_buffer=True)
    pred = pred_test.d
    pred = np.squeeze(pred, axis=1)
    pred = pred[0]
    pred = 2*(pred - np.min(pred))/np.ptp(pred)-1
    scipy.misc.imsave('stereo_depth.png', pred)

    print("Done")

   # Save NNP file (used in C++ inference later.).
   if args.save_nnp:
        runtime_contents = {
        'networks': [
            {'name': 'runtime',
             'batch_size': 1,
             'outputs': {'y0': pred_test},
             'names': {'x0': var_left, 'x1': var_right}}],
        'executors': [
            {'name': 'runtime',
             'network': 'runtime',
             'data': ['x0', 'x1'],
             'output': ['y0']}]}
        import nnabla.utils.save
        nnabla.utils.save.save(args.nnp, runtime_contents)


if __name__ == '__main__':
    main()
