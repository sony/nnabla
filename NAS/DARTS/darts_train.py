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

from __future__ import absolute_import
from six.moves import range

import os
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.initializer as I
import numpy as np
import json

from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

from args import get_args
from utils import *
from defined_functions import *
from cifar10_data import data_iterator_cifar10  # not good


def drop_path(x):
    """
        The same implementation as PyTorch versions.
        rate: Variable. drop rate. if the random value drawn from
                uniform distribution is less than the drop_rate,
                corresponding element becomes 0.
    """
    drop_prob = nn.parameter.get_parameter_or_create("drop_rate",
                                                     shape=(1, 1, 1, 1), need_grad=False)
    mask = F.rand(shape=(x.shape[0], 1, 1, 1))
    mask = F.greater_equal(mask, drop_prob)
    x = F.div2(x, 1 - drop_prob)
    x = F.mul2(x, mask)
    return x


def constructing_learned_cell(args, ops, arch_dict, which_cell,
                              cell_prev_prev, cell_prev, output_filter,
                              is_reduced_curr, is_reduced_prev, test):
    """
        Constructing one cell.
        input:
            args: arguments set by user.
            ops: operations used in the network.
            arch_dict: a dictionary containing architecture information.
            which_cell: int. An index of cell currently constructed.
            cell_prev_prev: Variable. Output of the cell behind the previous cell.
            cell_prev: Variable. Output of the previous cell.
            output_filter:t he number of the filter used for this cell.
            is_reduced_curr: bool. True if the current cell is the reduction cell.
            is_reduced_prev: bool. True if the previous cell is the reduction cell.
            test: bool. True if the network is for validation.
    """

    is_search = False  # Note that this is always False at retraining process.

    if is_reduced_curr:
        architecture = arch_dict["arch_reduction"]
        output_shape = (cell_prev.shape[0], output_filter,
                        cell_prev.shape[2] // 2, cell_prev.shape[3] // 2)
    else:
        architecture = arch_dict["arch_normal"]
        output_shape = (cell_prev.shape[0], output_filter,
                        cell_prev.shape[2], cell_prev.shape[3])

    if is_reduced_prev:
        scope = "fr{}".format(which_cell)
        cell_prev_prev = factorized_reduction(cell_prev_prev, output_filter,
                                              scope, test, is_search)
    else:
        scope = "preprocess_cell{}_node{}".format(which_cell, 0)
        cell_prev_prev = conv1x1(cell_prev_prev,
                                 output_filter, scope, test, is_search)

    scope = "preprocess_cell{}_node{}".format(which_cell, 1)
    cell_prev = conv1x1(cell_prev, output_filter, scope, test, is_search)

    nodes = [cell_prev_prev, cell_prev]
    num_of_nodes = args.num_nodes

    for j in range(2, num_of_nodes - 1):
        to_node = j
        current_arch = architecture[str(to_node)]  # [(op_id, input_node), ...]
        ops_ids = [op_id for op_id, _ in current_arch]
        from_nodes = [from_node for _, from_node in current_arch]

        input_0 = nodes[from_nodes[0]]
        ops_0 = ops[ops_ids[0]]
        scope_0 = "cell{}/node{}_{}/ops{}".format(
                        which_cell, from_nodes[0], to_node, ops_ids[0])
        input_0 = ops_0(input_0, output_filter, scope_0,
                        from_nodes[0], is_reduced_curr, test, is_search)
        if ops_0.__name__ != "identity" and not test:
            input_0 = drop_path(input_0)

        input_1 = nodes[from_nodes[1]]
        ops_1 = ops[ops_ids[1]]
        scope_1 = "cell{}/node{}_{}/ops{}".format(
                        which_cell, from_nodes[1], to_node, ops_ids[1])
        input_1 = ops_1(input_1, output_filter, scope_1,
                        from_nodes[1], is_reduced_curr, test, is_search)
        if ops_1.__name__ != "identity" and not test:
            input_1 = drop_path(input_1)

        intermediate_node = F.add2(input_0, input_1)
        nodes.append(intermediate_node)

    intermediate_nodes = nodes[2:num_of_nodes - 1]
    output = F.concatenate(*intermediate_nodes, axis=1)

    is_reduced_prev = is_reduced_curr
    return output, is_reduced_curr, is_reduced_prev, output_filter


def construct_networks(args, ops, arch_dict, image, test):
    """
        Construct a network by stacking cells.
        input:
            args: arguments set by user.
            ops: operations used in the network.
            arch_dict: a dictionary containing architecture information.
            image: Variable. Input images.
            test: bool. True if the network is for validation.
    """

    num_of_cells = args.num_cells
    initial_output_filter = args.output_filter + args.additional_filters_on_retrain

    num_class = 10
    aux_logits = None

    if not test:
        image = F.random_crop(F.pad(image, (4, 4, 4, 4)), shape=(image.shape))
        image = F.image_augmentation(image, flip_lr=True)
        image.need_grad = False
    x = image

    with nn.parameter_scope("stem_conv1"):
        stem_1 = PF.convolution(x, initial_output_filter,
                                (3, 3), (1, 1), with_bias=False)
        stem_1 = PF.batch_normalization(stem_1, batch_stat=not test)

    cell_prev, cell_prev_prev = stem_1, stem_1
    output_filter = initial_output_filter
    is_reduced_curr, is_reduced_prev = False, False

    for i in range(num_of_cells):
        if i in [num_of_cells // 3, 2*num_of_cells // 3]:
            output_filter = 2 * output_filter
            is_reduced_curr = True
        else:
            is_reduced_curr = False
        y, is_reduced_curr, is_reduced_prev, output_filter = \
            constructing_learned_cell(args, ops, arch_dict, i,
                                      cell_prev_prev, cell_prev, output_filter,
                                      is_reduced_curr, is_reduced_prev, test)

        if i == 2*num_of_cells // 3 and args.auxiliary and not test:
            print("Using Aux Tower after cell_{}".format(i))
            aux_logits = construct_aux_head(y, num_class)

        cell_prev, cell_prev_prev = y, cell_prev  # shifting

    y = F.average_pooling(y, y.shape[2:])  # works as global average pooling

    with nn.parameter_scope("fc"):
        pred = PF.affine(y, num_class, with_bias=True)

    return pred, aux_logits


def CNN_run(args, ops, arch_dict):
    """
        Based on the given model architecture,
        construct CNN and execute training.
        input:
            args: arguments set by user.
            ops: operations used in the network.
            arch_dict: a dictionary containing architecture information.
    """

    data_iterator = data_iterator_cifar10
    tdata = data_iterator(args.batch_size, True)
    vdata = data_iterator(args.batch_size, False)

    # CIFAR10 statistics, mean and variance
    CIFAR_MEAN = np.reshape([0.49139968, 0.48215827, 0.44653124], (1, 3, 1, 1))
    CIFAR_STD = np.reshape([0.24703233, 0.24348505, 0.26158768], (1, 3, 1, 1))

    channels, image_height, image_width = 3, 32, 32
    batch_size = args.batch_size
    initial_model_lr = args.model_lr

    one_epoch = tdata.size // batch_size
    max_iter = args.epoch * one_epoch
    val_iter = 10000 // batch_size

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
    pred_valid, _ = construct_networks(
        args, ops, arch_dict, image_valid, test=True)
    loss_valid = loss_function(pred_valid, label_valid)

    # set dropout rate in advance
    nn.parameter.get_parameter_or_create(
        "drop_rate", shape=(1, 1, 1, 1), need_grad=False)
    initial_drop_rate = nn.Variable((1, 1, 1, 1)).apply(d=args.dropout_rate)
    nn.parameter.set_parameter("drop_rate", initial_drop_rate)

    # prepare variables and graph used for training
    image_train = nn.Variable(
        (batch_size, channels, image_height, image_width))
    label_train = nn.Variable((batch_size, 1))
    input_image_train = {"image": image_train, "label": label_train}
    pred_train, aux_logits = construct_networks(
        args, ops, arch_dict, image_train, test=False)
    loss_train = loss_function(
        pred_train, label_train, aux_logits, args.auxiliary_weight)

    # prepare solvers
    model_params_dict = nn.get_parameters()
    solver_model = S.Momentum(initial_model_lr)
    solver_model.set_parameters(
        model_params_dict, reset=False, retain_state=True)

    # Training-loop
    for curr_epoch in range(args.epoch):
        print("epoch {}".format(curr_epoch))

        curr_dropout_rate = F.add_scalar(
            F.mul_scalar(initial_drop_rate, (curr_epoch / args.epoch)), 1e-8)
        nn.parameter.set_parameter("drop_rate", curr_dropout_rate)

        for i in range(one_epoch):
            image, label = tdata.next()
            image = image / 255.0
            image = (image - CIFAR_MEAN) / CIFAR_STD
            if args.cutout:
                image = cutout(image, args)
            input_image_train["image"].d = image
            input_image_train["label"].d = label
            loss_train.forward(clear_no_need_grad=True)

            e = categorical_error(pred_train.d, input_image_train["label"].d)
            monitor_loss.add(one_epoch*curr_epoch + i, loss_train.d.copy())
            monitor_err.add(one_epoch*curr_epoch + i, e)

            if args.lr_control_model:
                new_lr = learning_rate_scheduler(one_epoch*curr_epoch + i,
                                                 max_iter, initial_model_lr, 0)
                solver_model.set_learning_rate(new_lr)

            solver_model.zero_grad()
            loss_train.backward(clear_buffer=True)

            if args.with_grad_clip_model:
                for k, v in model_params_dict.items():
                    v.grad.copy_from(F.clip_by_norm(
                        v.grad, args.grad_clip_value_model))

            # update parameters
            solver_model.weight_decay(args.weight_decay_model)
            solver_model.update()

            if (one_epoch*curr_epoch+i) % args.model_save_interval == 0:
                nn.save_parameters(os.path.join(
                    args.model_save_path, 'params_{}.h5'.format(one_epoch*curr_epoch + i)))

        # Validation during training.
        ve = 0.
        vloss = 0.
        for j in range(val_iter):
            image, label = vdata.next()
            image = image / 255.0
            image = (image - CIFAR_MEAN) / CIFAR_STD
            input_image_valid["image"].d = image
            input_image_valid["label"].d = label
            loss_valid.forward(clear_no_need_grad=True)
            vloss += loss_valid.d.copy()
            ve += categorical_error(pred_valid.d.copy(), label)
        ve /= val_iter
        vloss /= val_iter
        monitor_vloss.add(one_epoch*curr_epoch+i, vloss)
        monitor_verr.add(one_epoch*curr_epoch+i, ve)

    return


def main():
    """
        Start architecture evaluation (retraining from scratch).
    """
    args = get_args()
    print(args)

    ctx = get_extension_context(args.context,
                                device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    ext = nn.ext_utils.import_extension_module(args.context)

    assert os.path.exists(
        args.model_arch_name), "architecture's params seem to be missing!"

    ops = {0: dil_conv_3x3, 1: dil_conv_5x5, 2: sep_conv_3x3, 3: sep_conv_5x5,
           4: max_pool_3x3, 5: avg_pool_3x3, 6: identity, 7: zero}

    with open(args.model_arch_name, 'r') as f:
        arch_dict = json.load(f)

    print("Train the model whose architecture is:")
    show_derived_cell(args, ops, arch_dict["arch_normal"], "normal")
    show_derived_cell(args, ops, arch_dict["arch_reduction"], "reduction")
    CNN_run(args, ops, arch_dict)

    return


if __name__ == '__main__':
    main()
