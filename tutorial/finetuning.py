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

import os
import importlib
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import numpy as np

from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.utils.nnp_graph import NnpLoader, NnpNetworkPass
from nnabla.utils.image_utils import imread
from nnabla.utils.data_iterator import data_iterator_simple

import argparse


def get_data_iterator_and_num_class(args):
    """
        Get Data_iterator for training and test data set.
        Also, obtain class / category information from data.
    """
    if args.train_csv:
        from nnabla.utils.data_iterator import data_iterator_csv_dataset
        data_iterator = data_iterator_csv_dataset

        if args.test_csv:
            assert os.path.isfile(
                args.test_csv), "csv file for test not found."

            # check the number of the classes / categories
            with open(args.train_csv, "r") as f:
                csv_data_train = f.readlines()[1:]  # line 1:"x:image,y:label"
            classes_train = {line.split(",")[-1].strip()
                             for line in csv_data_train}

            with open(args.test_csv, "r") as f:
                # first line:"x:image,y:label"
                csv_data_test = f.readlines()[1:]
            classes_test = {line.split(",")[-1].strip()
                            for line in csv_data_test}
            classes_train.update(classes_test)

            num_class = len(classes_train)

            data_iterator_train = data_iterator_csv_dataset(
                    args.train_csv, args.batch_size, args.shuffle,
                    normalize=False)

            data_iterator_valid = data_iterator_csv_dataset(
                    args.test_csv, args.batch_size, args.shuffle,
                    normalize=False)
        else:
            print("No csv file for test given. So split the training data")
            assert isintance(args.ratio, float), "ratio must be in (0.0, 1.0)"

            # check the number of the classes / categories
            with open(args.train_csv, "r") as f:
                # first line is "x:image,y:label"
                csv_data_train = f.readlines()[1:]
            all_classes = {line.split(",")[-1].strip()
                           for line in csv_data_train}
            num_class = len(all_classes)
            all_data = data_iterator_csv_dataset(
                    args.train_csv, args.batch_size, args.shuffle,
                    normalize=False)

            num_samples = all_data.size
            num_train_samples = int(args.ratio * num_samples)

            data_iterator_train = all_data.slice(
                rng=None, slice_start=0, slice_end=num_train_samples)

            data_iterator_valid = all_data.slice(
                rng=None, slice_start=num_train_samples, slice_end=num_samples)

    else:
        # use caltech101 data like tutorial
        from caltech101_data import data_iterator_caltech101
        assert isintance(args.ratio, float), "ratio must be in (0.0, 1.0)"
        data_iterator = data_iterator_caltech101
        num_class = 101  # pre-defined (excluding background class)
        all_data = data_iterator(
                args.batch_size, width=args.width, height=args.height)

        num_samples = all_data.size
        num_train_samples = int(args.ratio * num_samples)

        data_iterator_train = all_data.slice(
            rng=None, slice_start=0, slice_end=num_train_samples)

        data_iterator_valid = all_data.slice(
            rng=None, slice_start=num_train_samples, slice_end=num_samples)

    print("training images: {}".format(data_iterator_train.size))
    print("validation images: {}".format(data_iterator_valid.size))
    print("{} categories included.".format(num_class))

    return data_iterator_train, data_iterator_valid, num_class


def learning_rate_scheduler(curr_iter, T_max, eta_max, eta_min=0):
    """
        cosine annealing scheduler.
    """
    lr = eta_min + 0.5 * (eta_max - eta_min) * \
        (1 + np.cos(np.pi*(curr_iter / T_max)))
    return lr


def loss_function(pred, label):
    """
        Compute loss.
    """
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss


def construct_networks(args, images, model, num_class, test):
    try:
        pooled = model(images, force_global_pooling=1,
                       use_up_to="pool", training=not test)
    except:
        pooled = model(images, use_up_to="pool", training=not test)

    with nn.parameter_scope("finetuning"):
        if args.model == "VGG":
            pooled = F.relu(pooled)

            with nn.parameter_scope("additional_fc_1"):
                pooled = PF.affine(pooled, 4096)
            pooled = F.relu(pooled)
            if not test:
                pooled = F.dropout(pooled, 0.5)

            with nn.parameter_scope("additional_fc_2"):
                pooled = PF.affine(pooled, 4096)
            pooled = F.relu(pooled)
            if not test:
                pooled = F.dropout(pooled, 0.5)

        with nn.parameter_scope("last_fc"):
            pred = PF.affine(pooled, num_class)

    return pred


def CNN_run(args, model):

    data_iterator_train, data_iterator_valid, num_class = \
                get_data_iterator_and_num_class(args)

    channels, image_height, image_width = 3, args.height, args.width
    batch_size = args.batch_size
    initial_model_lr = args.model_lr

    one_epoch = data_iterator_train.size // batch_size
    max_iter = args.epoch * one_epoch
    val_iter = data_iterator_valid.size // batch_size

    # Create monitor.
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=100)
    monitor_err = MonitorSeries("Training error", monitor, interval=100)
    monitor_vloss = MonitorSeries("Test loss", monitor, interval=100)
    monitor_verr = MonitorSeries("Test error", monitor, interval=100)

    # prepare variables and graph used for test
    image_valid = nn.Variable(
        (batch_size, channels, image_height, image_width))
    label_valid = nn.Variable((batch_size, 1))
    input_image_valid = {"image": image_valid, "label": label_valid}

    pred_valid = construct_networks(
        args, image_valid, model, num_class, test=True)
    pred_valid.persistent = True
    loss_valid = loss_function(pred_valid, label_valid)
    top_1e_valid = F.mean(F.top_n_error(pred_valid, label_valid))

    # prepare variables and graph used for training
    image_train = nn.Variable(
        (batch_size, channels, image_height, image_width))
    label_train = nn.Variable((batch_size, 1))
    input_image_train = {"image": image_train, "label": label_train}

    pred_train = construct_networks(
        args, image_train, model, num_class, test=False)
    loss_train = loss_function(pred_train, label_train)
    top_1e_train = F.mean(F.top_n_error(pred_train, label_train))

    # prepare solvers
    solver = S.Momentum(initial_model_lr)
    solver.set_parameters(nn.get_parameters())

    # Training-loop
    for i in range(max_iter):
        image, label = data_iterator_train.next()
        input_image_train["image"].d = image
        input_image_train["label"].d = label
        nn.forward_all([loss_train, top_1e_train], clear_no_need_grad=True)

        monitor_loss.add(i, loss_train.d.copy())
        monitor_err.add(i, top_1e_train.d.copy())

        if args.lr_control_model:
            new_lr = learning_rate_scheduler(i, max_iter, initial_model_lr, 0)
            solver.set_learning_rate(new_lr)

        solver.zero_grad()
        loss_train.backward(clear_buffer=True)

        if args.with_grad_clip_model:
            for k, v in nn.get_parameters().items():
                v.grad.copy_from(F.clip_by_norm(
                    v.grad, args.grad_clip_value_model))

        # update parameters
        solver.weight_decay(args.weight_decay_model)
        solver.update()

        if i % args.model_save_interval == 0:
            # Validation during training.
            ve = 0.
            vloss = 0.
            for j in range(val_iter):
                v_image, v_label = data_iterator_valid.next()
                input_image_valid["image"].d = v_image
                input_image_valid["label"].d = v_label
                nn.forward_all([loss_valid, top_1e_valid], clear_buffer=True)
                vloss += loss_valid.d.copy()
                ve += top_1e_valid.d.copy()

            ve /= val_iter
            vloss /= val_iter
            monitor_vloss.add(i, vloss)
            monitor_verr.add(i, ve)

            nn.save_parameters(os.path.join(
               args.model_save_path, 'params_{}.h5'.format(i)))

    ve = 0.
    vloss = 0.
    for j in range(val_iter):
        v_image, v_label = data_iterator_valid.next()
        input_image_valid["image"].d = v_image
        input_image_valid["label"].d = v_label
        nn.forward_all([loss_valid, top_1e_valid], clear_buffer=True)
        vloss += loss_valid.d.copy()
        ve += top_1e_valid.d.copy()

    ve /= val_iter
    vloss /= val_iter
    monitor_vloss.add(i, vloss)
    monitor_verr.add(i, ve)

    nn.save_parameters(os.path.join(
       args.model_save_path, 'params_{}.h5'.format(i)))

    return


def main(args):

    if not args.train_csv:
        print("No user-made data given. Use caltech101 dataset for finetuning.")
    else:
        # prepare dataset.
        assert os.path.isfile(
            args.train_csv), "csv file for training not found, create dataset first."

    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    ext = nn.ext_utils.import_extension_module(args.context)

    print("Use {} for fine-tuning".format(args.model))
    model_name = args.model
    if model_name == "ResNet":
        num_layers = args.res_layer
    elif model_name == "VGG":
        num_layers = args.vgg_layer
    elif model_name == "SqueezeNet":
        num_layers = args.squeeze_ver
    else:
        num_layers = ""

    model_module = importlib.import_module("nnabla.models.imagenet")
    MODEL = getattr(model_module, model_name)
    if model_name in ["ResNet", "VGG", "SqueezeNet"]:
        model = MODEL(num_layers)  # got model
    else:
        model = MODEL()

    CNN_run(args, model)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model settings
    parser.add_argument('--model', type=str,
                        choices=["ResNet18",
                                 "ResNet34",
                                 "ResNet50",
                                 "ResNet101",
                                 "ResNet152",
                                 "MobileNet",
                                 "MobileNetV2",
                                 "SENet",
                                 "SqueezeNetV10",
                                 "SqueezeNetV11",
                                 "VGG11",
                                 "VGG13",
                                 "VGG16",
                                 "NIN",
                                 "DenseNet",
                                 "InceptionV3",
                                 "Xception",
                                 "GoogLeNet",
                                 "ResNet",
                                 "SqueezeNet",
                                 "VGG",
                                 ],
                        default="ResNet", help='name of the model')

    parser.add_argument('--height', type=int, default=128, help='image height')
    parser.add_argument('--width', type=int, default=128, help='image width')

    # Dataset settings
    parser.add_argument('--train-csv', type=str, default="",
                        help='.csv file which contains all the image path used for training')
    parser.add_argument('--test-csv', type=str, default="",
                        help='.csv file which contains all the image path used for training')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='ratio Training samples to Validation samples')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='whether or not to execute shuffe. Better to set it True.')

    # General settings
    parser.add_argument('--context', '-c', type=str, default='cudnn',
                        help="Extension path. ex) cpu, cudnn.")
    parser.add_argument("--device-id", "-d", type=str, default='0',
                        help='Device ID the training run on. \
                        This is only valid if you specify `-c cudnn`.')
    parser.add_argument("--type-config", "-t", type=str, default='float',
                        help='Type of computation. e.g. "float", "half".')
    parser.add_argument("--epoch", "-e", type=int, default=10)

    # batch_size
    parser.add_argument("--batch-size", "-b", type=int, default=32)

    # learning rate and its control
    parser.add_argument("--model-lr", type=float, default=0.025)
    parser.add_argument("--lr-control-model", type=bool, default=True,)

    # gradient clip model
    parser.add_argument("--with-grad-clip-model", type=bool, default=False)
    parser.add_argument("--grad-clip-value-model", type=float, default=5.0)

    # weight_decay
    parser.add_argument("--weight-decay-model", type=float, default=3e-4,
                        help='Weight decay rate. Weight decay is executed by default. \
                        Set it 0 to virtually disable it.')

    # misc
    parser.add_argument("--monitor-path", "-m",
                        type=str, default='tmp.monitor')
    parser.add_argument("--model-save-interval", "-s", type=int, default=1000)
    parser.add_argument("--model-save-path", "-o",
                        type=str, default='tmp.monitor')

    # DEPRECATED. model-specific arguments
    parser.add_argument('--res-layer', type=int, choices=[
                        18, 34, 50, 101, 152], default=18, help='DEPRECATED. which variation to use for ResNet')
    parser.add_argument('--vgg-layer', type=int,
                        choices=[11, 13, 16], default=16, help='DEPRECATED. which variation to use for VGG')
    parser.add_argument('--squeeze-ver', type=str, choices=[
                        'v1.0', 'v1.1'], default='v1.1', help='DEPRECATED. which version of SqueezeNet')
    args = parser.parse_args()

    main(args)
