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

import numpy as np

import os
import sys
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.initializer as I
from nnabla.ext_utils import get_extension_context

from cifar10_data import data_iterator_cifar10
from macro_CNN import CNN_run, get_data_stats, show_arch
from args import get_macro_args


def stack_lstm(x, prev_h, prev_c, state_size):
    """
        stacked LSTMs. Consists of 2 layers inside.
    """
    lstm_size = prev_h[0].shape[1]
    next_h = [nn.Variable([1, lstm_size], need_grad=True)
              for _ in range(len(prev_h))]
    next_c = [nn.Variable([1, lstm_size], need_grad=True)
              for _ in range(len(prev_c))]
    for layer_id, (_h, _c) in enumerate(zip(prev_h, prev_c)):
        inputs = x if layer_id == 0 else next_h[layer_id - 1]
        with nn.parameter_scope(str(layer_id)):
            curr_h, curr_c = PF.lstm(inputs, _h, _c, state_size)
        next_h[layer_id] = curr_h
        next_c[layer_id] = curr_c
    return next_h, next_c


def params_count(params):
    """
        count the number of the parameters.
    """
    def get_from_tuples(some_tuples):
        num = 1
        for t in some_tuples:
            num *= t
        return num
    nums = 0
    for v in params.values():
        nums += get_from_tuples(v.shape)
    return nums


def sample_from_controller(args):
    """
        2-layer RNN(LSTM) based controller which outputs an architecture of CNN, 
        represented as a sequence of integers and its list.
        Given the number of layers, for each layer, 
        it executes 2 types of computation, one for sampling the operation at that layer,
        another for sampling the skip connection patterns.
    """

    entropys = nn.Variable([1, 1], need_grad=True)
    log_probs = nn.Variable([1, 1], need_grad=True)
    skip_penaltys = nn.Variable([1, 1], need_grad=True)

    entropys.d = log_probs.d = skip_penaltys.d = 0.0  # initialize them all

    num_layers = args.num_layers
    lstm_size = args.lstm_size
    state_size = args.state_size
    lstm_num_layers = args.lstm_layers
    skip_target = args.skip_prob
    temperature = args.temperature
    tanh_constant = args.tanh_constant
    num_branch = args.num_ops

    arc_seq = []
    initializer = I.UniformInitializer((-0.1, 0.1))

    prev_h = [nn.Variable([1, lstm_size], need_grad=True)
              for _ in range(lstm_num_layers)]
    prev_c = [nn.Variable([1, lstm_size], need_grad=True)
              for _ in range(lstm_num_layers)]

    for i in range(len(prev_h)):
        prev_h[i].d = 0  # initialize variables in lstm layers.
        prev_c[i].d = 0

    inputs = nn.Variable([1, lstm_size])
    inputs.d = np.random.normal(0, 0.5, [1, lstm_size])

    g_emb = nn.Variable([1, lstm_size])
    g_emb.d = np.random.normal(0, 0.5, [1, lstm_size])

    skip_targets = nn.Variable([1, 2])
    skip_targets.d = np.array([[1.0 - skip_target, skip_target]])

    for layer_id in range(num_layers):
        # One-step stacked LSTM.
        with nn.parameter_scope("controller_lstm"):
            next_h, next_c = stack_lstm(inputs, prev_h, prev_c, state_size)
        prev_h, prev_c = next_h, next_c  # shape:(1, lstm_size)

        # Compute for operation.
        with nn.parameter_scope("ops"):
            logit = PF.affine(next_h[-1], num_branch,
                              w_init=initializer, with_bias=False)

        if temperature is not None:
            logit = F.mul_scalar(logit, (1 / temperature))

        if tanh_constant is not None:
            logit = F.mul_scalar(
                F.tanh(logit), tanh_constant)  # (1, num_branch)

        # normalizing logits.
        normed_logit = np.e ** logit.d
        normed_logit = normed_logit / np.sum(normed_logit)

        # Sampling operation id from multinomial distribution.
        ops_id = np.random.multinomial(1, normed_logit[0], 1).nonzero()[1]
        ops_id = nn.Variable.from_numpy_array(ops_id)  # (1, )
        arc_seq.append(ops_id.d)

        # log policy for operation.
        log_prob = F.softmax_cross_entropy(
            logit, F.reshape(ops_id, shape=(1, 1)))  # (1, )
        # accumulate log policy as log probs
        log_probs = F.add2(log_probs, log_prob)

        entropy = log_prob * F.exp(-log_prob)
        entropys = F.add2(entropys, entropy)  # accumulate entropy as entropys.

        w_emb = nn.parameter.get_parameter_or_create(
            "w_emb", [num_branch, lstm_size], initializer, need_grad=False)

        inputs = F.reshape(w_emb[int(ops_id.d)],
                           (1, w_emb.shape[1]))  # (1, lstm_size)

        with nn.parameter_scope("controller_lstm"):
            next_h, next_c = stack_lstm(inputs, prev_h, prev_c, lstm_size)
        prev_h, prev_c = next_h, next_c  # (1, lstm_size)

        with nn.parameter_scope("skip_affine_3"):
            adding_w_1 = PF.affine(
                next_h[-1], lstm_size, w_init=initializer, with_bias=False)  # (1, lstm_size)

        if layer_id == 0:
            inputs = g_emb  # (1, lstm_size)
            anchors = next_h[-1]  # (1, lstm_size)
            anchors_w_1 = adding_w_1  # then goes back to the entry point of the loop

        else:
            # (layer_id, lstm_size) this shape during the process
            query = anchors_w_1

            with nn.parameter_scope("skip_affine_1"):
                query = F.tanh(F.add2(query, PF.affine(
                    next_h[-1], lstm_size, w_init=initializer, with_bias=False)))
                #              (layer_id, lstm_size)   +   (1, lstm_size)
                # broadcast occurs here. resulting shape is; (layer_id, lstm_size)

            with nn.parameter_scope("skip_affine_2"):
                query = PF.affine(query, 1, w_init=initializer,
                                  with_bias=False)  # (layer_id, 1)
            # note that each weight for skip_affine_X is shared across all steps of LSTM.

            # re-define logits, now its shape is;(layer_id, 2)
            logit = F.concatenate(-query, query, axis=1)

            if temperature is not None:
                logit = F.mul_scalar(logit, (1 / temperature))

            if tanh_constant is not None:
                logit = F.mul_scalar(F.tanh(logit), tanh_constant)

            skip_prob_unnormalized = F.exp(logit)  # (layer_id, 2)

            # normalizing skip_prob_unnormalized.
            summed = F.sum(skip_prob_unnormalized, axis=1,
                           keepdims=True).apply(need_grad=False)
            summed = F.concatenate(summed, summed, axis=1)

            skip_prob_normalized = F.div2(
                skip_prob_unnormalized, summed)  # (layer_id, 2)

            # Sampling skip_pattern from multinomial distribution.
            skip_pattern = np.random.multinomial(
                1, skip_prob_normalized.d[0], layer_id).nonzero()[1]  # (layer_id, 1)
            arc_seq.append(skip_pattern)
            skip = nn.Variable.from_numpy_array(skip_pattern)

            # compute skip penalty.
            # (layer_id, 2) broadcast occurs here too
            kl = F.mul2(skip_prob_normalized, F.log(
                F.div2(skip_prob_normalized, skip_targets)))
            kl = F.sum(kl, keepdims=True)
            # get the mean value here in advance.
            kl = kl * (1.0 / (num_layers - 1))

            # accumulate kl divergence as skip penalty.
            skip_penaltys = F.add2(skip_penaltys, kl)

            # log policy for connection.
            log_prob = F.softmax_cross_entropy(
                logit, F.reshape(skip, shape=(skip.shape[0], 1)))
            log_probs = F.add2(log_probs, F.sum(log_prob, keepdims=True))

            entropy = F.sum(log_prob * F.exp(-log_prob), keepdims=True)
            # accumulate entropy as entropys.
            entropys = F.add2(entropys, entropy)

            skip = F.reshape(skip, (1, layer_id))

            inputs = F.affine(skip, anchors).apply(
                need_grad=False)  # (1, lstm_size)
            inputs = F.mul_scalar(inputs, (1.0 / (1.0 + (np.sum(skip.d)))))

            # add new row for the next computation
            # (layer_id + 1, lstm_size)
            anchors = F.concatenate(anchors, next_h[-1], axis=0)
            # (layer_id + 1, lstm_size)
            anchors_w_1 = F.concatenate(anchors_w_1, adding_w_1, axis=0)

    return arc_seq, log_probs, entropys, skip_penaltys


def get_sample_and_feedback(args, data_dict):
    """
        Let the controller predict one architecture and test its performance to get feedback.
        Here the feedback is validation accuracy and will be reused to train the controller. 
    """

    skip_weight = args.skip_weight
    entropy_weight = args.entropy_weight
    bl_dec = args.baseline_decay

    arc_seq, log_probs, entropys, skip_penaltys = sample_from_controller(args)

    sample_arch = list()
    for arc in arc_seq:
        sample_arch.extend(arc.tolist())

    show_arch(sample_arch)

    sample_entropy = entropys
    sample_log_prob = log_probs

    nn.set_auto_forward(False)
    val_acc = CNN_run(args, sample_arch, data_dict)  # Execute Evaluation Only
    nn.set_auto_forward(True)

    print("Accuracy on Validation: {:.2f} %\n".format(100*val_acc))

    reward = val_acc  # use validation accuracy as reward

    if entropy_weight is not None:
        reward = F.add_scalar(F.mul_scalar(
            sample_entropy, entropy_weight), reward).d

    sample_log_prob = F.mul_scalar(sample_log_prob, (1 / args.num_candidate))

    if args.use_variance_reduction:
        baseline = 0.0
        # variance reduction
        baseline = baseline - ((1 - bl_dec) * (baseline - reward))
        reward = reward - baseline

    loss = F.mul_scalar(sample_log_prob, (-1) * reward)

    if skip_weight is not None:
        adding_penalty = F.mul_scalar(skip_penaltys, skip_weight)
        loss = F.add2(loss, adding_penalty)

    return loss, val_acc, sample_arch


def sample_arch_and_train(args, data_dict, controller_weights_dict):
    """
        Execute these process.
        1. For a certain number of times, let the controller construct sample architectures 
           and test their performances. (By calling get_sample_and_feedback)
        2. By using the performances acquired by the previous process, train the controller.
        3. Select one architecture with the best validation accuracy and train its parameters.
    """

    solver = S.Momentum(args.control_lr)  # create solver for the controller
    solver.set_parameters(controller_weights_dict,
                          reset=False, retain_state=True)
    solver.zero_grad()

    val_list = list()
    arch_list = list()

    with nn.auto_forward():
        for c in range(args.num_candidate):
            output_line = " Architecture {} / {} ".format(
                (c + 1), args.num_candidate)
            print("{0:-^80s}".format(output_line))

            # sample one architecture and get its feedback for RL as loss
            loss, val_acc, sample_arch = get_sample_and_feedback(
                args, data_dict)

            val_list.append(val_acc)
            arch_list.append(sample_arch)
            loss.backward()  # accumulate gradient each time

        print("{0:-^80s}\n".format(" Reinforcement Learning Phase "))
        print("current accumulated loss:", loss.d)

        solver.weight_decay(0.025)
        solver.update()  # train the controller

        print("\n{0:-^80s}\n".format(" CNN Learning Phase "))
        best_idx = np.argmax(val_list)
        sample_arch = arch_list[best_idx]
        print("Train the model whose architecture is:")
        show_arch(sample_arch)
        print("and its accuracy is: {:.2f} %\n".format(100*np.max(val_list)))
        print("Learnable Parameters:", params_count(nn.get_parameters()))

    # train a child network which achieves the best validation accuracy.
    val_acc = CNN_run(args, sample_arch, data_dict, with_train=True)

    return sample_arch, val_acc


def search_architecture(args, data_dict, controller_weights_dict):
    """
        Execute architecture search. Keep searching until;
        it finds the architecture which achieves higher validation accuracy than the threshold set by users.
        it finishes the search for a certain number of times.
    """

    val_change = list()
    arch_change = list()
    best_val = 0.0

    for k in range(args.max_search_iter):
        output_line = " Iteration {} / {} ".format(
            (k + 1), args.max_search_iter)
        print("\n{0:-^80s}\n".format(output_line))

        if k > 3:
            print("Previous 3 Accuracy Changes:", "{:.2f}".format(100*val_change[-3]), "% ->",
                  "{:.2f}".format(100*val_change[-2]), "% ->",
                  "{:.2f}".format(100*val_change[-1]), "%\n")
            print("The best accuracy so far is: {:.2f} %.".format(
                np.max(val_change)*100))

        sample_arch, val_acc = sample_arch_and_train(
            args, data_dict, controller_weights_dict)
        arch_change.append(sample_arch)
        val_change.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_arch = sample_arch
            print(
                "Achieved the best validation accuracy. Saved the model architecture...")
            np.save(args.recommended_arch, np.array(best_arch))

            nn.save_parameters(os.path.join(
                    args.model_save_path, 'controller_params.h5'))

        if args.early_stop_over < best_val:
            print("Reached at Stop Accuracy. Finishes Architecture Search.")
            break

    print("During {0} trial, Best Accuracy is {1:.2f} %.".format(
        (k + 1), 100*np.max(val_change)))

    return arch_change, best_arch


def arguments_assertion(args):
    assert args.output_filter % 2 == 0, "Need even number of filters (for factorized reduction)."
    assert args.additional_filters_on_retrain % 2 == 0, "Need even number of additional filters."
    assert args.grad_clip_value > 0, "threshold must be positive number."


def sample_from_pretrained_controller(args):
    """
        Experimental Implementation.
    """
    assert args.num_sampling > 0, "num_sampling must be > 0."
    path = os.path.join(
            args.model_save_path, 'controller_params.h5')
    assert os.path.exists(path), "controller's weights seem to be missing!"
    nn.parameter.load_parameters(path)

    for i in range(args.num_sampling):
        output_line = " Sampled Architecture {} / {} ".format(
            (i + 1), args.num_sampling)
        print("\n{0:-^80s}\n".format(output_line))

        with nn.auto_forward():
            arc_seq, _, _, _ = sample_from_controller(args)

        sample_arch = list()
        for arc in arc_seq:
            sample_arch.extend(arc.tolist())

        show_arch(sample_arch)
        filename = "sampled_macro_arch_{}.npy".format(i)
        np.save("sampled_macro_arch_{}.npy".format(i), np.array(sample_arch))

    print("when you want to train the sampled network from scratch,\n\
    type like 'python macro_retrain.py <OPTION> --recommended-arch {}'".format(filename))


def main():
    """
        Start architecture search and save the architecture found by the controller during the search.
    """
    args = get_macro_args()
    arguments_assertion(args)

    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)
    ext = nn.ext_utils.import_extension_module(args.context)

    if args.sampling_only:
        sample_from_pretrained_controller(args)
        return

    data_iterator = data_iterator_cifar10
    tdata = data_iterator(args.batch_size, True)
    vdata = data_iterator(args.batch_size, False)
    mean_val_train, std_val_train, channel, img_height, img_width, num_class = get_data_stats(
        tdata)
    mean_val_valid, std_val_valid, _, _, _, _ = get_data_stats(vdata)

    data_dict = {"train_data": (tdata, mean_val_train, std_val_train),
                 "valid_data": (vdata, mean_val_valid, std_val_valid),
                 "basic_info": (channel, img_height, img_width, num_class)}

    initializer = I.UniformInitializer((-0.1, 0.1))

    # Prepare all the weights in advance
    controller_weights_and_shape = {
        'controller_lstm/0/lstm/affine/W': (2 * args.lstm_size, 4, args.lstm_size),
        'controller_lstm/0/lstm/affine/b': (4, args.lstm_size),
        'controller_lstm/1/lstm/affine/W': (2 * args.lstm_size, 4, args.lstm_size),
        'controller_lstm/1/lstm/affine/b': (4, args.lstm_size),
        'ops/affine/W': (args.lstm_size, args.num_ops),
        'skip_affine_1/affine/W': (args.lstm_size, args.lstm_size),
        'skip_affine_2/affine/W': (args.lstm_size, 1),
        'skip_affine_3/affine/W': (args.lstm_size, args.lstm_size)}
    for w_name, w_shape in controller_weights_and_shape.items():
        nn.parameter.get_parameter_or_create(
            w_name, w_shape, initializer=initializer, need_grad=True)

    # create dictionary of controller's weights
    controller_weights_dict = {w_name: nn.get_parameters(
    )[w_name] for w_name in controller_weights_and_shape.keys()}

    arch_change, best_arch = search_architecture(
        args, data_dict, controller_weights_dict)

    if args.select_strategy == "best":
        print("saving the model which achieved the best validation accuracy as {}.".format(
            args.recommended_arch))
        check_arch = best_arch
    else:
        # Use the latest architecture. it's not necessarily the one with the best architecture.
        print("saving the latest model recommended by the controller as {}.".format(
            args.recommended_arch))
        check_arch = arch_change[-1]
        np.save(args.recommended_arch, np.array(check_arch))

    print("The saved architecture is;")
    show_arch(check_arch)
    print("when you want to train the network from scratch,\n\
    type 'python macro_retrain.py <OPTION> --recommended-arch {}'".format(args.recommended_arch))

    # save the controller's weights so that another architectures can be made.
    all_params = nn.get_parameters(grad_only=False)
    controller_weights = list(controller_weights_and_shape.keys()) + ["w_emb"]
    for param_name in all_params.keys():
        if param_name not in controller_weights:
            nn.parameter.pop_parameter(param_name)
    nn.save_parameters(os.path.join(
            args.model_save_path, 'controller_params.h5'))

    # If you want to train the model recommended by the controller from scratch
    # right after architecture search, uncomment the lines below
    # nn.clear_parameters()
    # ext.clear_memory_cache()  # clear all the Variables
    # val_acc = CNN_run(args, check_arch, data_dict, with_train=True, after_search=True)
    return


if __name__ == '__main__':
    main()
