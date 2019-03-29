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
import csv
from nnabla.utils.image_utils import imsave

parser = argparse.ArgumentParser(description='psm_net')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='max disparity')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./psmnet_trained_param_10.h5',
                    help='load model')
parser.add_argument('--savemodel', default="./backup_psm_net_kitti",
                    help='save model')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--batchsize_train', type=int, default=4,
                    help='batch size train')
parser.add_argument('--batchsize_test', type=int, default=2,
                    help='batch size test')
parser.add_argument('--crop_width', type=int, default=512,
                    help='set cropped image width')
parser.add_argument('--crop_height', type=int, default=256,
                    help='set cropped image height')
parser.add_argument('--im_width', type=int, default=1232,
                    help='set image width')
parser.add_argument('--im_height', type=int, default=368,
                    help='set  image height')
parser.add_argument('-g', '--gpus', type=str,
                    help='GPU IDs to be used', default="4")
parser.add_argument('--dataset', type=str,
                    help='select dataset from "SceneFlow" or "Kitti"', default="Kitti")
args = parser.parse_args()


def read_csv(path):
    f = open(path, "r")
    reader = csv.reader(f)
    header = next(reader, None)
    column = {}
    for h in header:
        column[h] = []
    for row in reader:
        for k, v in zip(header, row):
            column[k].append(v)
    return column["L"], column["R"], column["disp"]


def adjust_learning_rate(epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print(lr, "learning rate")
    return lr


def main():

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare for CUDA.
    ctx = get_extension_context('cudnn', device_id=args.gpus)
    nn.set_default_context(ctx)

    start_full_time = time.time()
    from iterator import data_iterator

    # Data list for sceneflow data set
    train_list = "./dataset/kitti_train.csv"
    test_list = "./dataset/kitti_test.csv"

    train = True
    validation = False

    # Set monitor path.
    monitor_path = './nnmonitor' + str(datetime.now().strftime("%Y%m%d%H%M%S"))

    img_left, img_right, disp_img = read_csv(train_list)
    img_left_test, img_right_test, disp_img_test = read_csv(test_list)
    train_samples = len(img_left)
    test_samples = len(img_left_test)
    train_size = int(len(img_left) / args.batchsize_train)
    test_size = int(len(img_left_test) / args.batchsize_test)

    # Create data iterator.
    data_iterator_train = data_iterator(
        train_samples, args.batchsize_train, img_left, img_right, disp_img, train, shuffle=True, dataset=args.dataset)
    data_iterator_test = data_iterator(
        test_samples, args.batchsize_test, img_left_test, img_right_test, disp_img_test, validation, shuffle=False, dataset=args.dataset)
    # Set data size
    print(train_size, test_size)
    # Clrear patameters
    nn.clear_parameters()
    # Define data shape for training.
    var_left = nn.Variable(
        (args.batchsize_train, 3, args.crop_height, args.crop_width))
    var_right = nn.Variable(
        (args.batchsize_train, 3, args.crop_height, args.crop_width))
    var_disp = nn.Variable(
        (args.batchsize_train, 1, args.crop_height, args.crop_width))
    # Define data shape for testing.
    var_left_test = nn.Variable(
        (args.batchsize_test, 3, args.im_height, args.im_width))
    var_right_test = nn.Variable(
        (args.batchsize_test, 3, args.im_height, args.im_width))
    var_disp_test = nn.Variable(
        (args.batchsize_test, 1, args.im_height, args.im_width))
    if args.loadmodel is not None:
        # Loading CNN pretrained parameters.
        nn.load_parameters(args.loadmodel)
    # === for Training ===
    # Definition of pred
    pred1, pred2, pred3 = psm_net(var_left, var_right, args.maxdisp, True)
    mask_train = F.greater_scalar(var_disp, 0)
    sum_mask = F.maximum_scalar(F.sum(mask_train), 1)
    print(sum_mask.d, "sum_mask_first")
    # Definition of loss
    loss = 0.5 * (0.5 * F.sum(F.huber_loss(pred1, var_disp)*mask_train)/(sum_mask) + 0.7 * F.sum(F.huber_loss(
        pred2, var_disp)*mask_train)/(sum_mask) + F.sum(F.huber_loss(pred3, var_disp)*mask_train)/(sum_mask))
    # === for Testing ===
    # Definition of pred
    pred_test = psm_net(var_left_test, var_right_test, args.maxdisp, False)
    var_gt = var_disp_test + F.less_equal_scalar(var_disp_test, 0) * -1
    var_pred = pred_test + F.less_equal_scalar(pred_test, 0) * -1
    E = F.abs(var_pred - var_gt)
    n_err = F.sum(F.logical_and(F.logical_and(F.greater_scalar(var_gt, 0.0), F.greater_scalar(E, 3.0)),
                                F.greater_scalar(F.div2(E, F.abs(var_gt)), 0.05)))
    n_total = F.sum(F.greater_scalar(var_gt, 0))
    test_loss = F.div2(n_err, n_total)
    # Prepare monitors.
    monitor = Monitor(monitor_path)
    monitor_train = MonitorSeries('Training loss', monitor, interval=1)
    monitor_test = MonitorSeries('Validation loss', monitor, interval=1)
    monitor_time_train = MonitorTimeElapsed(
        "Training time/epoch", monitor, interval=1)
    # Create a solver (parameter updater)
    solver = S.Adam(alpha=0.001, beta1=0.9, beta2=0.999)
    # Set Parameters
    params = nn.get_parameters()
    solver.set_parameters(params)
    params2 = nn.get_parameters(grad_only=False)
    solver.set_parameters(params2)
    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0
        index = 0
        lr = adjust_learning_rate(epoch)
        ###Training###
        while index < train_size:
            # Get mini batch
            # Preprocess
            var_left.d, var_right.d, var_disp.d = data_iterator_train.next()
            loss.forward(clear_no_need_grad=True)
            # Initialize gradients
            solver.zero_grad()
            # Backward execution
            loss.backward(clear_buffer=True)
            # Update parameters by computed gradients
            solver.set_learning_rate(lr)
            solver.update()
            print('Iter %d training loss = %.3f' % (index, loss.d))
            total_train_loss += loss.d
            index += 1
        train_error = total_train_loss/train_size
        print('epoch %d total training loss = %.3f' % (epoch, train_error))
        monitor_time_train.add(epoch)
        #  ## teting ##
        total_test_loss = 0
        max_acc = 0
        index_test = 0
        while index_test < test_size:
            var_left_test.d, var_right_test.d, var_disp_test.d = data_iterator_test.next()
            test_loss.forward(clear_buffer=True)
            total_test_loss += test_loss.d
            print('Iter %d test loss = %.3f' % (index_test, test_loss.d*100))
            index_test += 1
        test_error = total_test_loss/test_size
        print('epoch %d total 3-px error in val = %.3f' %
              (epoch, test_error*100))
        if test_error > max_acc:
            max_acc = test_error*100
        print('MAX epoch %d total test error = %.3f' % (epoch, max_acc))
        # Pass validation loss to a monitor.
        monitor_test.add(epoch, test_error*100)
        # Pass training loss to a monitor.
        monitor_train.add(epoch, train_error)
        print('full training time = %.2f HR' %
              ((time.time() - start_full_time)/3600))
        # Save Parameter
        out_param_file = os.path.join(
            args.savemodel, 'psmnet_trained_param_' + str(epoch) + '.h5')
        nn.save_parameters(out_param_file)


if __name__ == '__main__':
    main()
