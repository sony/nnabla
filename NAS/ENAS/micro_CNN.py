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
    local_used_weights = set()
    if elem in [0, 1, 2, 3]:
        # pooling layers. They only have 1x1 conv layer.
        local_used_weights.update({"{}/1x1conv/conv/W".format(scope),
                                   "{}/1x1conv/bn/gamma".format(scope),
                                   "{}/1x1conv/bn/beta".format(scope)})
    if elem in [0, 1]:
        # depthwise_separable conv
        local_used_weights.update({"{}/conv/W".format(scope),
                                   "{}/bn/gamma".format(scope),
                                   "{}/bn/beta".format(scope),
                                   "{}/depthwise_conv/W".format(scope)})

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


def identity(x, output_filter, scope, test):
    """
        Identity function.
    """
    return F.identity(x)


def conv1x1(x, output_filter, scope, test):
    """
        1x1 convolution, works more like a constant multiplier.
    """
    C = x.shape[1]
    with nn.parameter_scope("1x1conv"):
        h = PF.convolution(x, C, (1, 1), with_bias=False)
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
        path2 = F.slice(path2, (0, 0, 1, 1))
        path2 = F.average_pooling(path2, (1, 1), (2, 2))
        with nn.parameter_scope("path2_conv"):
            path2 = PF.convolution(
                path2, output_filter // 2, (1, 1), with_bias=False)

        final_path = F.concatenate(path1, path2, axis=1)
        with nn.parameter_scope("reduction_bn"):
            final_path = PF.batch_normalization(
                final_path, batch_stat=not test)

    return final_path


def construct_cell(prev_layer, architecture, num_nodes, output_filter, scope, test):
    """
    Constructing the cell, which consists of multiple nodes.
    The cell made by this function is either Conv cell or Reduction cell.
    layer_id is somewhere between [0, num_cells)
    the name of the incoming scope should be "w{}".format(layer_id), like "w2" at layer 2 
    so in the end scope name will be like w1_<node>_<ops>
    """
    assert len(architecture) // 4 == num_nodes
    used_indices = set()
    local_used_weights = set()

    ops = {0: depthwise_separable_conv3x3, 1: depthwise_separable_conv5x5,
           2: average_pool, 3: max_pool, 4: identity}

    # 2 previous outputs to be fed as inputs
    layers = [prev_layer[-2], prev_layer[-1]]

    for node in range(num_nodes):
        ind = node
        # get ops id and input index
        one_node = architecture[4*ind:4*(ind + 1)]
        idx_1, ops_1, idx_2, ops_2 = one_node
        # store the node's index used as input
        used_indices.update({idx_1, idx_2})
        scope_1 = "{0}_{1}_{2}".format(scope, node, ops_1)
        scope_2 = "{0}_{1}_{2}".format(scope, node, ops_2)

        # for each nodes, apply the operation and get its weights name
        h1 = ops[ops_1](layers[idx_1], output_filter, scope_1, test)
        local_used_weights.update(get_weights_name(scope_1, ops_1))

        h2 = ops[ops_2](layers[idx_2], output_filter, scope_2, test)
        local_used_weights.update(get_weights_name(scope_2, ops_2))

        # add them as output of that node
        h_add = F.add2(h1, h2)
        layers.append(h_add)  # store each output temporarily

    all_indices = set(range(num_nodes + 2))  # all nodes in the cell
    # exclude nodes not used as others' input
    candidates = all_indices - used_indices
    h_out = layers[candidates.pop()]  # randomly pop output

    for j in candidates:
        h_out = F.add2(h_out, layers[j])  # repeatedly sum up outputs

    return h_out, local_used_weights


def get_reference_layers(num_cells, pool_layers):
    """
        By analyzing the network, get 2 lists named ref_groups and required_indices.
        each element of ref_groups is the number of downsampling 
        applied to the output of the layer, for example, if ref_groups[i] = 1, 
        at ith layer, its input shape should be 1x downsampled shape
        compared to the original spatial size. 
        (in short, if the spatial size of the data is 32x32, input at the layer i is 16x16).
        also, it's better to know if the output is reused at the following layers
        in a different shape (especially after a reduction cell). 
        If so, the output needs to be downsampled in advance. 
        required_indices tells you if this advance downsampling needs to be applied or not.
    """

    ref_groups = num_cells * [0]  # (stem_cells + other cells)
    x_list = pool_layers + [len(ref_groups) - 1]
    index = 0

    for n in range(len(ref_groups)):
        if n <= x_list[index]:
            ref_groups[n] = index
        else:
            index += 1
            ref_groups[n] = index

    # ref_groups' index (N) indicates that
    # Nth layer is pooled (spatially reduced) for N times.
    # for example, 0 means no pooling is applied to that layer.
    ref_groups = [0, 0] + ref_groups  # first 2 zeros are for stem cells.

    # we also need to know how many times we need to downsample the output for the node 1 step ahead
    ref_groups_next = ref_groups[1:] + [ref_groups[-1]]
    # same for the node 2 step ahead
    ref_groups_next_next = ref_groups[2:] + [ref_groups[-1], ref_groups[-1]]

    # we now know if we need to change the spatial size of output for each cell
    required_indices = [set(group) for group in zip(
        ref_groups, ref_groups_next, ref_groups_next_next)]

    return ref_groups, required_indices


def construct_architecture(image, num_class, num_cells, num_nodes, both_archs, output_filter, test):
    """
        Construct an architecture based on the given lists.
        Note that first 2 layers are stem conv and have nothing to do with node operations.
    """
    conv_arch, reduc_arch = both_archs

    aux_logits = None
    used_weights = set()

    pool_distance = num_cells // 3
    pool_layers = [pool_distance - 1, 2*pool_distance - 1]
    pool_layers = [_ for _ in pool_layers if _ > 0]

    if len(pool_layers) > 0:
        aux_head_indices = [pool_layers[-1] + 1]
    else:
        # this must not be happened. since num_cells needs to be more than 3.
        aux_head_indices = [1]

    ref_groups, required_indices = get_reference_layers(num_cells, pool_layers)
    prev_layers = [list() for _ in range(ref_groups[-1] + 1)]

    # Note that this implementation is slightly different from the one written by tensorflow.
    if not test:
        image = F.image_augmentation(
            image, angle=0.25, flip_lr=True)  # random_crop, min_scale
        image.need_grad = False
    x = image

    # --------------------------------------- 1st cell ---------------------------------------
    with nn.parameter_scope("stem_conv1"):
        x = PF.convolution(x, output_filter, (3, 3), (1, 1), with_bias=False)
        x = PF.batch_normalization(x, batch_stat=not test)
    used_weights.update(
        {"stem_conv1/conv/W", "stem_conv1/bn/gamma", "stem_conv1/bn/beta"})
    prev_layers[0].append(x)  # store to the "unpooled" layer

    # spatial reduction (this might be skipped)
    for i in range(1, len(required_indices[0])):
        curr_scope = "stem1_reduc{}".format(i)
        x = factorized_reduction(x, 2*x.shape[1], curr_scope, test)

        local_used_weights = get_factorized_weights_name(curr_scope)
        used_weights.update(local_used_weights)
        prev_layers[i].append(x)

    # --------------------------------------- 2nd cell ---------------------------------------
    with nn.parameter_scope("stem_conv2"):
        x = PF.convolution(
            prev_layers[0][-1], output_filter, (3, 3), (1, 1), with_bias=False)
        x = PF.batch_normalization(x, batch_stat=not test)
    used_weights.update(
        {"stem_conv2/conv/W", "stem_conv2/bn/gamma", "stem_conv2/bn/beta"})
    prev_layers[0].append(x)  # store to the "unpooled" layer

    # spatial reduction (this might be skipped)
    for i in range(1, len(required_indices[1])):
        curr_scope = "stem2_reduc{}".format(i)
        x = factorized_reduction(x, 2*x.shape[1], curr_scope, test)

        local_used_weights = get_factorized_weights_name(curr_scope)
        used_weights.update(local_used_weights)
        prev_layers[i].append(x)

    # ------------------------------- Normal / Reduction cells -------------------------------
    for layer_id in range(2, num_cells):
        using_layer_index = ref_groups[layer_id]
        required_index = list(required_indices[layer_id])
        required_index.sort()
        scope = 'w{}'.format(layer_id)

        if layer_id in pool_layers:
            architecture = reduc_arch
        else:
            architecture = conv_arch

        previous_outputs = prev_layers[using_layer_index]
        x, local_used_weights = construct_cell(
            previous_outputs, architecture, num_nodes, previous_outputs[-1].shape[1], scope, test)
        used_weights.update(local_used_weights)
        prev_layers[using_layer_index].append(x)

        required_index.remove(using_layer_index)  # discard an index used above

        # if this output (x) is reused as an input in other cells and
        # its shape needs to be changed, apply downsampling in advance
        for i in required_index:
            curr_scope = "scope{0}_reduc{1}".format(layer_id, i)
            x = factorized_reduction(x, 2*x.shape[1], curr_scope, test)
            local_used_weights = get_factorized_weights_name(curr_scope)
            used_weights.update(local_used_weights)
            prev_layers[i].append(x)

        # auxiliary head, to use the intermediate output for training
        if layer_id in aux_head_indices and not test:
            print("Using aux_head at layer {}".format(layer_id))
            aux_logits = F.relu(x)
            aux_logits = F.average_pooling(aux_logits, (5, 5), (3, 3))

            with nn.parameter_scope("proj"):
                aux_logits = PF.convolution(
                    aux_logits, 128, (3, 3), (1, 1), with_bias=False)
                aux_logits = PF.batch_normalization(
                    aux_logits, batch_stat=not test)
                aux_logits = F.relu(aux_logits)
            used_weights.update(
                {"proj/conv/W", "proj/bn/gamma", "proj/bn/beta"})

            with nn.parameter_scope("avg_pool"):
                aux_logits = PF.convolution(
                    aux_logits, 768, (3, 3), (1, 1), with_bias=False)
                aux_logits = PF.batch_normalization(
                    aux_logits, batch_stat=not test)
                aux_logits = F.relu(aux_logits)
            used_weights.update(
                {"avg_pool/conv/W", "avg_pool/bn/gamma", "avg_pool/bn/beta"})

            with nn.parameter_scope("additional_fc"):
                aux_logits = F.global_average_pooling(aux_logits)
                aux_logits = PF.affine(aux_logits, num_class, with_bias=False)
            used_weights.update({"additional_fc/affine/W"})

    x = F.global_average_pooling(prev_layers[-1][-1])

    if not test:
        dropout_rate = 0.5
        x = F.dropout(x, dropout_rate)

    with nn.parameter_scope("fc"):
        pred = PF.affine(x, num_class, with_bias=False)
        used_weights.add("fc/affine/W")

    return pred, aux_logits, used_weights


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


def loss_function(pred, aux_logits, label):
    """
        Compute loss.
    """
    if aux_logits is None:
        loss = F.mean(F.softmax_cross_entropy(pred, label))
    else:
        loss = F.softmax_cross_entropy(pred, label)
        loss_from_aux = F.softmax_cross_entropy(aux_logits, label)
        loss = F.mean(F.add2(loss, loss_from_aux))
    return loss


def show_arch(both_archs):
    """
    Showing architectures of the cell.
    Note that architectures shown by this is just a part of the entire network.
    """
    conv_arch, reduc_arch = both_archs
    print(conv_arch)
    print(reduc_arch)


def learning_rate_scheduler(curr_iter, T_max, eta_max, eta_min=0):
    """
        Cosine annealing scheduler.
    """
    lr = eta_min + 0.5 * (eta_max - eta_min) * \
        (1 + np.cos(np.pi*(curr_iter / T_max)))
    return lr


def CNN_run(args, both_archs, data_dict, with_train=False, after_search=False):
    """
    """

    num_cells = args.num_cells
    num_nodes = args.num_nodes

    if after_search:
        assert with_train is True, "when you train the network after architecture search, set with_train=True"
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

    image_valid = nn.Variable(
        (batch_size, channels, image_height, image_width))
    input_image_valid = {"image": image_valid}

    vdata._reset()  # rewind data

    test = True
    pred_valid, _, _ = construct_architecture(image_valid, num_class, num_cells, num_nodes,
                                              both_archs, output_filter, test)

    if with_train:
        if after_search:
            # setting for training after architecture search
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
        pred_train, aux_logits, used_weights = construct_architecture(image_train, num_class, num_cells, num_nodes,
                                                                      both_archs, output_filter, test)
        loss_train = loss_function(pred_train, aux_logits, label_train)

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
            if i > 0 and i % one_epoch == 0:
                # Validation during training.
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
