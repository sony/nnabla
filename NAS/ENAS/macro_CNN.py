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
import numpy as np

from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed


def get_weights_name(scope, elem):
    """
        get weight names used for usual operations.
    """
    local_used_weights = {"{}/1x1conv/conv/W".format(scope),
                          "{}/1x1conv/bn/gamma".format(scope),
                          "{}/1x1conv/bn/beta".format(scope)}
    if elem in [0, 1, 2, 3]:
        # at least normal conv
        local_used_weights.update({"{}/conv/W".format(scope),
                                   "{}/bn/gamma".format(scope),
                                   "{}/bn/beta".format(scope)})
    if elem in [2, 3]:
        # depthwise_separable conv
        local_used_weights.add("{}/depthwise_conv/W".format(scope))

    return local_used_weights


def get_factorized_weights_name(scope):
    """
        get weight names used for factorized_reduction.
    """
    local_used_weights = {"{}/path1_conv/conv/W".format(scope),
                          "{}/path2_conv/conv/W".format(scope),
                          "{}/reduction_bn/bn/gamma".format(scope),
                          "{}/reduction_bn/bn/beta".format(scope)}
    return local_used_weights


def conv3x3(x, output_filter, scope, test):
    """
        convolution with kernel 3x3.
    """
    with nn.parameter_scope(scope):
        h = conv1x1(x, output_filter, scope, test)
        h = PF.convolution(h, output_filter, (3, 3), (1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def conv5x5(x, output_filter, scope, test):
    """
        convolution with kernel 5x5.
    """
    with nn.parameter_scope(scope):
        h = conv1x1(x, output_filter, scope, test)
        h = PF.convolution(h, output_filter, (5, 5), (2, 2), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def depthwise_separable_conv3x3(x, output_filter, scope, test):
    """
        depthwise separable convolution with kernel 3x3.
    """
    with nn.parameter_scope(scope):
        h = conv1x1(x, output_filter, scope, test)
        h = PF.depthwise_convolution(h, (3, 3), (1, 1), with_bias=False)
        h = PF.convolution(h, output_filter, (1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def depthwise_separable_conv5x5(x, output_filter, scope, test):
    """
        depthwise separable convolution with kernel 5x5.
    """
    with nn.parameter_scope(scope):
        h = conv1x1(x, output_filter, scope, test)
        h = PF.depthwise_convolution(h, (5, 5), (2, 2), with_bias=False)
        h = PF.convolution(h, output_filter, (1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def max_pool(x, output_filter, scope, test):
    """
        max pooling (with no spatial downsampling).        
    """
    with nn.parameter_scope(scope):
        h = conv1x1(x, output_filter, scope, test)
    h = F.max_pooling(h, (3, 3), (1, 1), pad=(1, 1))
    return h


def average_pool(x, output_filter, scope, test):
    """
        average pooling (with no spatial downsampling).        
    """
    with nn.parameter_scope(scope):
        h = conv1x1(x, output_filter, scope, test)
    h = F.average_pooling(h, (3, 3), (1, 1), pad=(1, 1))
    return h


def conv1x1(x, output_filter, scope, test):
    """
        1x1 convolution, works more like a constant multiplier.
    """
    with nn.parameter_scope("1x1conv"):
        h = PF.convolution(x, output_filter, (1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test)
        h = F.relu(h)
    return h


def categorical_error(pred, label):
    """
    Compute categorical error given score vectors and labels as
    numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def factorized_reduction(x, output_filter, scope, test):
    """
        Applying spatial reduction to input variable.
        Input variable is passed to:
        Skip path 1, applied average pooling with stride 2.
        Skip path 2, first padded with 0 on the right and bottom, 
                     then shifted by 1 (so that those 0-padded sides will be added, 
                     whereas its shape is the same as the original),
        Then these 2 variables are concatenated along the depth dimension.
    """
    with nn.parameter_scope(scope):
        path1 = F.average_pooling(x, (1, 1), (2, 2))
        with nn.parameter_scope("path1_conv"):
            path1 = PF.convolution(
                path1, output_filter // 2, (1, 1), with_bias=False)

        path2 = F.pad(x, (0, 1, 0, 1), mode='constant')
        path2 = path2[:, :, 1:, 1:]
        path2 = F.average_pooling(path2, (1, 1), (2, 2))
        with nn.parameter_scope("path2_conv"):
            path2 = PF.convolution(
                path2, output_filter // 2, (1, 1), with_bias=False)

        final_path = F.concatenate(path1, path2, axis=1)
        with nn.parameter_scope("reduction_bn"):
            final_path = PF.batch_normalization(
                final_path, batch_stat=not test)

    return final_path


def apply_ops_and_connect(x, prev_layers, connect_pattern, ops, elem, output_filter, scope, test):
    """
        execute the operation at the current layer.
        and if there is a skip connection with the previous layers,
        sum the whole values up. 
    """

    assert len(prev_layers) == len(connect_pattern) + 1

    is_skip_connected = False  # set as False initially.
    local_used_weights = set()
    # feeding previous output to the current layer.
    x = ops[elem](x, output_filter, scope, test)
    local_used_weights.update(get_weights_name(scope, elem))

    for flag, prev_output in zip(connect_pattern, prev_layers[:-1]):
        # ignore the last variable stored in prev_layers[-1]
        # since that is the previous layer's output (which is already used as the input above)
        if flag == 1:
            # skip connection exists.
            is_skip_connected = True
            x = F.add2(x, prev_output)

    if is_skip_connected:
        with nn.parameter_scope(scope + "/skip"):
            x = PF.batch_normalization(x, batch_stat=not test)
        local_used_weights.update(
            {"{}/skip/bn/gamma".format(scope), "{}/skip/bn/beta".format(scope)})

    return x, local_used_weights


def arc_to_matrix(connect_patterns):
    """
        Creates matrix which represents the connection info between layers.
        if the matrix[i,j] = 1, (i - 1)th and jth layers are connected. 
        note that (-1)st layer means stem conv cell.
    """
    mat_size = len(connect_patterns)
    # use triangular matrix.
    connection_matrix = np.tri(mat_size, dtype=np.int8)
    for i in range(mat_size):
        connect_pattern = connect_patterns[i]
        connection_matrix[i, :len(connect_pattern)] = connect_pattern
    return connection_matrix


def get_requirement_strict(ref_groups, connect_patterns, pool_layers):
    """
    It tells you which variables and pooled variables you need to prepare(1) or not(0). 
    Returns a list named required_indices. for example;
         initial   1x pooled    2x pooled             initial   1x pooled    2x pooled
          size        size         size                size        size         size
    array([[1,          0,           0],         array([[1,          1,           0],
           [1,          0,           0],                [1,          1,           1],
           [0,          1,           0],                [0,          1,           0],
           [0,          1,           0],                [0,          1,           1],
           [0,          0,           1],                [0,          0,           1],
           [0,          0,           1],                [0,          0,           1],

    why need this:
    we need to know if we should prepare spatially-reduced (pooled) versions
    of the variable from the current layer, since some layers could be skip-connected 
    to distant layers whose output variables have different shape.
    To check it, first gets the connection matrix (see arc_to_matrix for more info),
    checks how each layer connects, and based on that get the required_indices.
    """

    same_size_ranges = [0] + [idx +
                              1 for idx in pool_layers] + [len(connect_patterns)]
    # output of layers from same_size_ranges[:, i:i+1] have the same spatial size

    connection_matrix = arc_to_matrix(connect_patterns)
    connection_matrix = connection_matrix.transpose()
    required_indices = list()
    for j in range(len(pool_layers) + 1):
        same_size_start = same_size_ranges[j]
        same_size_end = same_size_ranges[j + 1]
        # note that we treat True as 1 here.
        required_indices.append(
            (connection_matrix[:, same_size_start:same_size_end].sum(axis=1) > 0)*1)
    required_indices = np.array(required_indices)
    required_indices = required_indices.transpose()

    # we need intermediate downsampled variables. (ex. 64x32x32 -> 64x16x16 -> 64x8x8)
    for indices in required_indices:
        indices[np.where(indices == 1)[0][0]:np.where(indices == 1)[0][-1]] = 1

    # and one more intermediate downsampled variables.
    for pool_layer_id in pool_layers:
        required_indices[pool_layer_id, ref_groups[pool_layer_id + 1]] = 1

    return required_indices


def get_requirement_soft(ref_groups):
    """
    Works almost the same as get_requirement_strict, but this is roughly implemented.
    The returned indices are not always actually required. Some might be included in surplus.

    It tells you which variable and factorized-reduced variable you need to prepare.
         initial   1x pooled    2x pooled
          size        size         size
    array([[1,          1,           1],
           [1,          1,           1],
           [0,          1,           1],
           [0,          1,           1],
           [0,          0,           1],
           [0,          0,           1],

    In short, in one row (layer), if Nx pooled size are needed, 
    all the other Mx pooled size are considered it is required also (N < M).
    it's NOT likely to be memory efficient.
    """
    required_indices = np.zeros(
        [ref_groups[-1] + 1, len(ref_groups)], dtype=np.uint8)
    for j in range(len(ref_groups)):
        ind = ref_groups[j]
        required_indices[ind:, j] = 1
    required_indices = required_indices.transpose()
    return required_indices


def construct_architecture(image, num_class, operations, output_filter, test, connect_patterns):
    """
        Architecture Construction. 
    """
    ops = {0: conv3x3, 1: conv5x5, 2: depthwise_separable_conv3x3,
           3: depthwise_separable_conv5x5, 4: max_pool, 5: average_pool}

    used_weights = set()

    pool_distance = len(operations) // 3
    pool_layers = [pool_distance - 1, 2*pool_distance - 1]
    # exclude negative indices
    pool_layers = [idx for idx in pool_layers if idx > 0]

    ref_groups = len(operations) * [0]
    tmp_list = pool_layers + [len(operations) - 1]
    index = 0
    for n in range(len(operations)):
        if n <= tmp_list[index]:
            ref_groups[n] = index
        else:
            index += 1
            ref_groups[n] = index

    # elements in ref_groups tell you how many times you need to do pooling.
    # e.g. [0, 0, 0, 1, 1, 1, ..., 2] : the 1st layer needs no pooling,
    # but the last needs 2 poolings, to get spatially reduced variables.

    #required_indices = get_requirement_soft(ref_groups)
    required_indices = get_requirement_strict(
        ref_groups, connect_patterns, pool_layers)

    num_of_pooling = len(pool_layers)
    normal_layers = [list()]
    pooled_layers = [list() for j in range(num_of_pooling)]

    prev_layers = normal_layers + pooled_layers
    # prev_layer consists of: [[initial_size_layers], [1x pooled_layers], [2x pooled_layers], ...]

    if not test:
        image = F.image_augmentation(image, angle=0.25, flip_lr=True)
        image.need_grad = False
    x = image

    # next comes the basic operation. for the first layer,
    # just apply a convolution (to make the size of the input the same as that of successors)

    with nn.parameter_scope("stem_conv"):
        x = PF.convolution(x, output_filter, (3, 3), (1, 1), with_bias=False)
        x = PF.batch_normalization(x, batch_stat=not test)
        used_weights.update(
            {"stem_conv/conv/W", "stem_conv/bn/gamma", "stem_conv/bn/beta"})
    prev_layers[0].append(x)
    # "unpooled" variable is stored in normal_layers (prev_layers[0]).

    # then apply factorized reduction (kind of pooling),
    # but ONLY IF the spatially-reduced variable is required.
    # for example, when this layer has skip connection with latter layers.

    for j in range(1, len(prev_layers)):
        if required_indices[0][j]:
            nested_scope = "stem_pool_{}".format(j)
            reduced_var = factorized_reduction(
                prev_layers[j - 1][-1], output_filter, nested_scope, test)
            used_weights.update(get_factorized_weights_name(nested_scope))
        else:
            # dummy variable. Should never be used.
            reduced_var = nn.Variable([1, 1, 1, 1])
        prev_layers[j].append(reduced_var)

    # reduced (or "pooled") variable is stored in pooled_layers (prev_layers[1:]).

    # basically, repeat the same process, for whole layers.
    for i, elem in enumerate(operations):
        scope = 'w{}_{}'.format(i, elem)

        # basic operation (and connects it with previous layers if it has skip connections)

        using_layer_index = ref_groups[i]
        connect_pattern = connect_patterns[i]
        x, local_used_weights = apply_ops_and_connect(prev_layers[using_layer_index][-1],
                                                      prev_layers[using_layer_index], connect_pattern, ops, elem, output_filter, scope, test)

        used_weights.update(local_used_weights)
        prev_layers[using_layer_index].append(x)

        # factorized reduction

        for j in range(using_layer_index + 1, len(prev_layers)):
            if required_indices[i + 1][j]:
                nested_scope = "{0}_pool{1}".format(scope, j)
                reduced_var = factorized_reduction(
                    prev_layers[j - 1][-1], output_filter, nested_scope, test)
                used_weights.update(get_factorized_weights_name(nested_scope))
            else:
                reduced_var = nn.Variable([1, 1, 1, 1])  # dummy variable.
            prev_layers[j].append(reduced_var)

    x = F.global_average_pooling(x)

    if not test:
        dropout_rate = 0.5
        x = F.dropout(x, dropout_rate)

    with nn.parameter_scope("fc"):
        pred = PF.affine(x, num_class, with_bias=False)
        used_weights.add("fc/affine/W")

    return pred, used_weights


def get_data_stats(data, verbose=False):
    """
        Data Preprocessing.
        1. Divide each value by 255.
        2. Get the mean value and standard deviation.
        3. Later these values will be used,
           so that the input mean value becomes 0 
           and input standard deviation becomes 1.
    """
    max_iter = data.size // data.batch_size
    mean_sum = np.zeros([1, 3, 1, 1])
    var_sum = np.zeros([1, 3, 1, 1])
    min_index = 10000
    max_index = -10000
    for i in range(max_iter):
        image, label = data.next()

        if label.max() > max_index:
            max_index = label.max()

        if label.min() < min_index:
            min_index = label.min()

        image = image / 255.0
        mean = np.mean(image, axis=(0, 2, 3), keepdims=True)
        variance = np.var(image, axis=(0, 2, 3), keepdims=True)
        mean_sum += mean
        var_sum += variance

    mean_sum /= max_iter
    var_sum /= max_iter
    var_sum = np.sqrt(var_sum)
    num_channels, img_height, img_width = image.shape[1:]
    num_class = max_index - min_index + 1
    data._reset()
    if verbose:
        print("mean value:\n", mean_sum)
        print("standard deviation:\n", var_sum)
    return mean_sum, var_sum, num_channels, img_height, img_width, num_class


def loss_function(pred, label):
    """
        Compute loss.
    """
    loss = F.mean(F.softmax_cross_entropy(pred, label))
    return loss


def get_architecture(sample_arc):
    """
        Given the sample_arc, it returns 2 lists 
        which represent operations and connections at each layer.
    """
    operations = list()
    connect_patterns = list()
    start_idx = 0
    tmp = len(sample_arc)
    # mathematically we can get the num_layers
    num_layers = int(-1 + np.sqrt(1+8*tmp)) // 2

    for i in range(num_layers):
        operations.append(sample_arc[start_idx])
        connect_patterns.append(
            list(sample_arc[start_idx + 1:start_idx + i + 1]))
        start_idx = i + start_idx + 1

    return operations, connect_patterns


def show_arch(sample_arc):
    """
    Given the list containing the architecture, shows a full architecture of the network.
    """
    start_idx = 0
    tmp_str = " "
    tmp = len(sample_arc)
    num_layers = int(-1 + np.sqrt(1+8*tmp)) // 2
    margin = len(str(num_layers))
    for i in range(num_layers):
        ops = sample_arc[start_idx:start_idx + i + 1]
        print("layer", tmp_str*(margin - len(str(i))), i,
              ": ops:", ops[0], ", connection:", ops[1:])
        start_idx = i + start_idx + 1


def learning_rate_scheduler(curr_iter, T_max, eta_max, eta_min=0):
    """
        cosine annealing scheduler.
    """
    lr = eta_min + 0.5 * (eta_max - eta_min) * \
        (1 + np.cos(np.pi*(curr_iter / T_max)))
    return lr


def CNN_run(args, sample_arc, data_dict, with_train=False, after_search=False):
    """
        Based on the given architecture, constructs CNN and computes validation score.
        If passed with_train=True, it also executes CNN training.
        Set after_search=True only after the architecture search is finished.
        When after_search=True, some of the arguments change (e.g. epochs, output_filters)
    """
    if after_search:
        assert with_train is True, "when you retrain the network, set with_train=True"
    tdata, mean_val_train, std_val_train = data_dict["train_data"]
    vdata, mean_val_valid, std_val_valid = data_dict["valid_data"]
    channels, image_height, image_width, num_class = data_dict["basic_info"]
    batch_size = args.batch_size

    output_filter = args.output_filter

    if with_train:
        if after_search:
            num_epoch = args.epoch_on_retrain
            if args.additional_filters_on_retrain > 0:
                output_filter += args.additional_filters_on_retrain
        else:
            num_epoch = args.epoch_per_search

        one_epoch = tdata.size // batch_size
        max_iter = num_epoch * one_epoch

    val_iter = args.val_iter

    monitor_path = args.monitor_path
    model_save_path = args.monitor_path
    decay_rate = args.weight_decay
    initial_lr = args.child_lr

    model_save_interval = args.model_save_interval

    ops = {0: conv3x3, 1: conv5x5, 2: depthwise_separable_conv3x3,
           3: depthwise_separable_conv5x5, 4: max_pool, 5: average_pool}
    # actual operations can be called by: ops[index](input, output_filter, scope)

    operations, connect_patterns = get_architecture(sample_arc)

    # prepare variables used for validation
    image_valid = nn.Variable(
        (batch_size, channels, image_height, image_width))
    input_image_valid = {"image": image_valid}

    vdata._reset()  # rewind data

    test = True
    pred_valid, _ = construct_architecture(image_valid, num_class,
                                           operations, output_filter, test, connect_patterns)

    if with_train:
        if after_search:
            # setting for when retraining
            with_grad_clip = args.with_grad_clip_on_retrain
            grad_clip = args.grad_clip_value
            lr_control = args.lr_control_on_retrain
        else:
            with_grad_clip = args.with_grad_clip_on_search
            grad_clip = args.grad_clip_value
            lr_control = args.lr_control_on_search

        # prepare variables used for training
        image_train = nn.Variable(
            (batch_size, channels, image_height, image_width))
        label_train = nn.Variable((batch_size, 1))
        input_image_train = {"image": image_train, "label": label_train}

        tdata._reset()  # rewind data

        test = False
        pred_train, used_weights = construct_architecture(image_train, num_class, operations,
                                                          output_filter, test, connect_patterns)
        loss_train = loss_function(pred_train, label_train)

        used_weights_dict = {key_name: nn.get_parameters(
        )[key_name] for key_name in used_weights}

        # Create monitor.
        monitor = Monitor(monitor_path)
        monitor_loss = MonitorSeries("Training loss", monitor, interval=100)
        # modified to display accuracy.
        monitor_err = MonitorSeries("Training accuracy", monitor, interval=100)
        # modified to display accuracy.
        monitor_verr = MonitorSeries("Test accuracy", monitor, interval=1)

        # Solvers
        solver = S.Momentum(initial_lr)
        solver.set_parameters(
            used_weights_dict, reset=False, retain_state=True)

        # Training-loop
        for i in range(max_iter):

            # Validation during training.
            if i > 0 and i % one_epoch == 0:
                ve = 0.
                for j in range(val_iter):
                    image, label = vdata.next()
                    image = image / 255.0
                    image = (image - mean_val_valid) / std_val_valid
                    input_image_valid["image"].d = image
                    pred_valid.forward()
                    ve += categorical_error(pred_valid.d, label)
                ve /= val_iter
                monitor_verr.add(i, 1.0 - ve)  # modified to display accuracy.

            if after_search and int(i % args.model_save_interval) == 0:
                nn.save_parameters(os.path.join(
                    args.model_save_path, 'params_%06d.h5' % i))

            # Forward/Zerograd/Backward
            image, label = tdata.next()
            image = image / 255.0
            image = (image - mean_val_train) / std_val_train
            input_image_train["image"].d = image
            input_image_train["label"].d = label
            loss_train.forward()

            if lr_control:
                new_lr = learning_rate_scheduler(i, max_iter, initial_lr, 0)
                solver.set_learning_rate(new_lr)

            solver.zero_grad()
            loss_train.backward()

            if with_grad_clip:
                for k, v in used_weights_dict.items():
                    if np.linalg.norm(v.g) > grad_clip:
                        v.grad.copy_from(F.clip_by_norm(v.grad, grad_clip))

            # Solvers update
            solver.weight_decay(decay_rate)
            solver.update()
            e = categorical_error(pred_train.d, input_image_train["label"].d)
            monitor_loss.add(i, loss_train.d.copy())
            monitor_err.add(i, 1.0 - e)  # modified to display accuracy.

    # Validation (After training or when called for evaluation only)
    ve = 0.
    for j in range(val_iter):
        image, label = vdata.next()
        image = image / 255.0
        image = (image - mean_val_valid) / std_val_valid
        input_image_valid["image"].d = image
        pred_valid.forward()
        ve += categorical_error(pred_valid.d, label)
    ve /= val_iter

    if with_train:
        print("Validation Accuracy on Trained CNN:",
              '{:.2f}'.format(100*(1.0 - ve)), "%\n")

    if after_search:
        nn.save_parameters(os.path.join(
            args.model_save_path, 'params_%06d.h5' % (max_iter)))

    return 1.0 - ve
