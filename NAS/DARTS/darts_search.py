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
from cifar10_data import data_iterator_cifar10


def store_gradient(accumulated_gradient, alphas_dict, coeff=1.):
    """
        Storing weights' gradients into dictionary.
        Need to be called after loss.backward(),
        i.e. after each Variable gets its gradient.
    """
    accumulated_gradient = {
        k: accumulated_gradient[k]+coeff*alphas_dict[k].grad for k in alphas_dict.keys()}
    return accumulated_gradient


def weight_modify(original_weights, delta_gradient_w, model_params_dict, coeff=1.):
    """
        Modifies weights MANUALLY, when updating architecture parameters
        using second-order approximation.
        (For detailed information, plaese refer to the paper ).
        Unlike solver.update(), which automatically modifies the weights
        assocaiated to that solver, this function manually modifies weights.
    """
    for k, v in model_params_dict.items():
        nn.parameter.set_parameter(
            k, nn.Variable(v.shape, need_grad=True).apply(
                data=original_weights[k].data + (coeff*delta_gradient_w[k].data)))
    return


def constructing_cell(args, ops, which_cell, cell_prev_prev, cell_prev, output_filter,
                      is_reduced_curr, is_reduced_prev, test=False):
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

    # If True, all the parameters in batch_normalizations won't be updated.
    is_search = True

    if is_reduced_curr:
        keyname_basis = "alpha_reduction"
        output_shape = (cell_prev.shape[0], output_filter,
                        cell_prev.shape[2] // 2, cell_prev.shape[3] // 2)
    else:
        keyname_basis = "alpha_normal"
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

    num_of_nodes = args.num_nodes

    # latter_nodes are all the intermediate nodes,
    # except for 2 input nodes and 1 output node.
    latter_nodes = [nn.Variable(output_shape)
                    for _ in range(num_of_nodes - 2 - 1)]
    for v in latter_nodes:
        v.d = 0  # initialize.

    num_of_ops = len(ops)

    # prepare a list to store all nodes.
    nodes = [cell_prev_prev, cell_prev] + latter_nodes
    for i in range(num_of_nodes - 2):
        successors = [_ for _ in range(i + 1, num_of_nodes - 1)]
        for j in successors:
            if j == 1:
                continue
            from_node, to_node = i, j
            scope = "cell{}/node{}_{}".format(which_cell, from_node, to_node)

            stacked_x = num_of_ops*(nodes[i], )
            stacked_x = tuple([op(x, output_filter, scope + "/ops{}".format(op_id), i, is_reduced_curr, test, is_search)
                               for x, op, op_id in zip(stacked_x, tuple(ops.values()), tuple(ops.keys()))])
            y = F.stack(*stacked_x, axis=0)

            alpha_name = keyname_basis + "_{}_{}".format(i, j)
            current_alpha = nn.parameter.get_parameter_or_create(
                alpha_name, (num_of_ops,) + (1, 1, 1, 1))
            alpha_prob = F.softmax(current_alpha, axis=0)
            y = F.mul2(y, alpha_prob)
            if i == 0:
                nodes[j] = F.sum(y, axis=0)
            else:
                nodes[j] = F.add2(nodes[j], F.sum(y, axis=0))

    intermediate_nodes = nodes[2:num_of_nodes - 1]
    output = F.concatenate(*intermediate_nodes, axis=1)

    is_reduced_prev = is_reduced_curr
    return output, is_reduced_curr, is_reduced_prev, output_filter


def construct_networks(args, ops, image, test):
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
    initial_output_filter = args.output_filter

    num_class = 10
    stem_multiplier = 3

    if not test:
        image = F.random_crop(F.pad(image, (4, 4, 4, 4)), shape=(image.shape))
        image = F.image_augmentation(image, flip_lr=True)
        image.need_grad = False
    x = image

    with nn.parameter_scope("stem_conv1"):
        stem_1 = PF.convolution(
            x, initial_output_filter*stem_multiplier, (3, 3), (1, 1), with_bias=False)
        stem_1 = PF.batch_normalization(
            stem_1, batch_stat=not test)

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
            constructing_cell(args, ops, i,
                              cell_prev_prev, cell_prev, output_filter,
                              is_reduced_curr, is_reduced_prev)

        cell_prev, cell_prev_prev = y, cell_prev  # shifting

    y = F.average_pooling(y, y.shape[2:])  # works as global average pooling

    with nn.parameter_scope("fc"):
        pred = PF.affine(y, num_class, with_bias=True)

    return pred


def CNN_run(args, ops, alphas_dict):
    """
        Based on the given model architecture,
        construct CNN and execute training.
        input:
            args: arguments set by user.
            ops: operations used in the network.
            arch_dict: a dictionary containing architecture information.
    """

    data_iterator = data_iterator_cifar10
    all_data = data_iterator(args.batch_size, True)
    tdata = all_data.slice(rng=None, slice_start=0, slice_end=25000)
    vdata = all_data.slice(rng=None, slice_start=25000, slice_end=50000)

    # CIFAR10 statistics, mean and variance
    CIFAR_MEAN = np.reshape([0.49139968, 0.48215827, 0.44653124], (1, 3, 1, 1))
    CIFAR_STD = np.reshape([0.24703233, 0.24348505, 0.26158768], (1, 3, 1, 1))

    channels, image_height, image_width = 3, 32, 32
    batch_size = args.batch_size
    initial_model_lr = args.model_lr

    one_epoch = tdata.size // batch_size
    max_iter = args.epoch * one_epoch

    # Create monitor.
    monitor = Monitor(args.monitor_path)
    monitor_loss = MonitorSeries("Training loss", monitor, interval=100)
    monitor_err = MonitorSeries("Training error", monitor, interval=100)
    monitor_vloss = MonitorSeries("Validation loss", monitor, interval=100)
    monitor_verr = MonitorSeries("Validation error", monitor, interval=100)

    # prepare variables and graph used for training
    image_train = nn.Variable(
        (batch_size, channels, image_height, image_width))
    label_train = nn.Variable((batch_size, 1))
    input_image_train = {"image": image_train, "label": label_train}
    pred_train = construct_networks(args, ops, image_train, test=False)
    loss_train = loss_function(pred_train, label_train)

    # prepare solvers for model parameters
    model_params_dict = \
        {k: v for k, v in nn.get_parameters().items() if "alpha_" not in k}
    solver_model = S.Momentum(initial_model_lr)
    solver_model.set_parameters(
        {k: v for k, v in nn.get_parameters().items()
         if k in model_params_dict.keys()}, reset=False, retain_state=True)

    # prepare solvers for architecture parameters
    solver_archs = S.Adam(alpha=args.arch_lr, beta1=0.5, beta2=0.999)
    solver_archs.set_parameters(
        {k: v for k, v in nn.get_parameters().items()
         if k in alphas_dict.keys()}, reset=False, retain_state=True)

    # Training-loop
    for i in range(max_iter):

        # Update Model Parameters.

        if args.second_order:
            # store the weights before update.
            original_weights = {k: nn.Variable(v.shape, need_grad=True).apply(data=nn.NdArray(v.shape).copy_from(v.data))
                                for k, v in nn.get_parameters().items() if "alpha_" not in k}

            # gradients refuge
            accumulated_gradient = \
                {k: nn.Variable(v.shape).apply(d=0)
                 for k, v in alphas_dict.items()}

        image, label = tdata.next()
        image = image / 255.0
        image = (image - CIFAR_MEAN) / CIFAR_STD
        input_image_train["image"].d = image
        input_image_train["label"].d = label
        loss_train.forward()

        e = categorical_error(pred_train.d, input_image_train["label"].d)
        monitor_loss.add(i, loss_train.d.copy())
        monitor_err.add(i, e)

        if args.lr_control_model:
            new_lr = learning_rate_scheduler(i, max_iter, initial_model_lr, 0)
            solver_model.set_learning_rate(new_lr)

        solver_model.zero_grad()
        loss_train.backward(clear_buffer=True)

        if args.with_grad_clip_model:
            for k, v in model_params_dict.items():
                v.grad.copy_from(F.clip_by_norm(
                    v.grad, args.grad_clip_value_model))

        solver_model.weight_decay(args.weight_decay_model)
        solver_model.update()  # weights update ( w -> w')

        if args.second_order:
            updated_weights = {k: nn.Variable(v.shape, need_grad=True).apply(
                                    data=nn.NdArray(v.shape).copy_from(v.data))
                               for k, v in nn.get_parameters().items()
                               if "alpha_" not in k}

        # Update Architecture Parameters.

        ve, vloss = 0., 0.
        v_image, v_label = vdata.next()
        v_image = v_image / 255.0
        v_image = (v_image - CIFAR_MEAN) / CIFAR_STD
        input_image_train["image"].d = v_image
        input_image_train["label"].d = v_label
        # compute Loss_on_valid(w', alpha)
        loss_train.forward(clear_no_need_grad=True)

        ve = categorical_error(pred_train.d, input_image_train["label"].d)
        monitor_vloss.add(i, loss_train.d.copy())
        monitor_verr.add(i, ve)

        solver_archs.zero_grad()
        solver_model.zero_grad()
        loss_train.backward(clear_buffer=True)  # its gradient is stored

        if args.second_order:
            accumulated_gradient = store_gradient(
                accumulated_gradient, alphas_dict, coeff=1.)

            # grad_alpha_L_val(w', alpha).  Note that gradient stored into .data
            delta_gradient_w = {k: nn.Variable(v.shape).apply(data=nn.NdArray(v.shape).copy_from(v.grad), need_grad=True)
                                for k, v in nn.get_parameters().items() if "alpha_" not in k}

            epsilon = 0.01 / np.sum([np.linalg.norm(v.d)
                                     for v in delta_gradient_w.values()])

            coeff = 1.0*epsilon
            # w -> w+ (= w + epsilon*grad_Loss_on_val(w', alpha))
            weight_modify(original_weights, delta_gradient_w,
                          model_params_dict, coeff)

            input_image_train["image"].d = image  # reuse the same data
            input_image_train["label"].d = label

            # compute Loss_on_train(w+, alpha)
            loss_train.forward()
            solver_archs.zero_grad()
            solver_model.zero_grad()
            loss_train.backward(clear_buffer=True)  # its gradient is stored

            # accumulate currently registered gradient
            coeff = (-1.)*args.eta / 2.*epsilon
            accumulated_gradient = store_gradient(
                accumulated_gradient, alphas_dict, coeff)

            coeff = -1.0*epsilon
            # w -> w- (= w - epsilon*grad_Loss_on_val(w', alpha))
            weight_modify(original_weights, delta_gradient_w,
                          model_params_dict, coeff)

            # compute Loss_on_train(w-, alpha)
            loss_train.forward()
            solver_archs.zero_grad()
            solver_model.zero_grad()
            loss_train.backward(clear_buffer=True)  # its gradient is stored

            # accumulate currently registered gradient again
            coeff = (+1.)*args.eta / 2.*epsilon
            accumulated_gradient = store_gradient(
                accumulated_gradient, alphas_dict, coeff)

            # replace the weights
            for k, v in alphas_dict.items():
                nn.parameter.set_parameter(k, nn.Variable(v.shape).apply(
                        data=v.data, grad=accumulated_gradient[k], need_grad=True))
            for k, v in model_params_dict.items():
                nn.parameter.set_parameter(k, nn.Variable(v.shape).apply(
                        data=updated_weights[k].data, need_grad=True))

        solver_archs.weight_decay(args.weight_decay_archs)
        solver_archs.update()

        if i % 1000 == 0:
            for k, v in alphas_dict.items():
                keynames = k.split("_")
                print("\nParameters for {} cell, node {} to {};".format(
                    keynames[1], keynames[2], keynames[3]))
                show_ops_and_prob(v.d, ops)

    return alphas_dict


def main():
    """
        Start architecture search.
    """
    args = get_args()
    print(args)

    ctx = get_extension_context(args.context,
                                device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    ext = nn.ext_utils.import_extension_module(args.context)

    ops = {0: dil_conv_3x3, 1: dil_conv_5x5, 2: sep_conv_3x3, 3: sep_conv_5x5,
           4: max_pool_3x3, 5: avg_pool_3x3, 6: identity, 7: zero}

    initializer = I.UniformInitializer((-0.1, 0.1))
    num_of_nodes = args.num_nodes

    alphas_dict = dict()
    w_shape = (len(ops),) + (1, 1, 1, 1)

    # prepare architecture parameters in advance
    for i in range(num_of_nodes):
        for j in range(i + 1, num_of_nodes - 1):
            if j < 2:
                continue  # no connection exists between 1st and 2nd nodes.
            else:
                w_name_normal = "alpha_normal_{}_{}".format(i, j)
                w_name_reduction = "alpha_reduction_{}_{}".format(i, j)
                alphas_dict[w_name_normal] = \
                    nn.parameter.get_parameter_or_create(w_name_normal,
                                                         w_shape, initializer)
                alphas_dict[w_name_reduction] = \
                    nn.parameter.get_parameter_or_create(w_name_reduction,
                                                         w_shape, initializer)

    # run architecture search
    alphas_dict = CNN_run(args, ops, alphas_dict)
    for k in nn.get_parameters(grad_only=False).keys():
        if "alpha_" not in k:
            nn.parameter.pop_parameter(k)  # delete unnecessary parameters.

    print("Architecture Search is finished. The saved architecture is,")
    alpha_normal, alpha_reduction = arrange_weights(args, ops)
    arch_normal = parse_weights(args, alpha_normal)
    arch_reduction = parse_weights(args, alpha_reduction)
    show_derived_cell(args, ops, arch_normal, "normal")
    show_derived_cell(args, ops, arch_reduction, "reduction")

    arch_data = {"arch_normal": arch_normal, "arch_reduction": arch_reduction}
    print("Saving the architecture parameter: {}/{}".format(
                                    args.monitor_path, args.model_arch_name))
    model_path = args.model_arch_name
    with open(model_path, 'w') as f:
        json.dump(arch_data, f)

    print("when you want to train the network from scratch\n\
    type 'python darts_train.py <OPTION> \
    --monitor-path {} --model-arch-name {}".format(
                                    args.monitor_path, args.model_arch_name))

    return


if __name__ == '__main__':
    main()
