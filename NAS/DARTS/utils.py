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
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


def softmax(x):
    """
        just a softmax by numpy implementation.
    """
    y = np.exp(x) / np.sum(np.exp(x))
    y = np.reshape(y, (1, x.shape[0]))
    return y


def show_ops_and_prob(x, ops):
    # showing current chosen probability of each operation.
    for op in ops.values():
        print("{0: ^12s}".format(op.__name__), end=", ")
    print()
    y = softmax(x)
    for score in y[0]:
        str_score = str(score*100)[:4] + "%"
        print("{0: ^12s}".format(str_score), end=", ")
    print()
    return


def arrange_weights(args, ops):
    num_edges = ((args.num_nodes - 1) * (args.num_nodes - 2) // 2) - 1
    inshape = (1, len(ops))

    alpha_normal = {(i + 2): np.empty(((i + 2), len(ops)))
                    for i in range(args.num_nodes - 3)}
    alpha_reduction = {(i + 2): np.empty(((i + 2), len(ops)))
                       for i in range(args.num_nodes - 3)}

    for k, param in nn.get_parameters().items():
        from_node = int(k.split("_")[-2])
        to_node = int(k.split("_")[-1])

        if "_normal" in k:
            alpha_normal[to_node][from_node] = param.d.reshape(inshape)
        else:
            alpha_reduction[to_node][from_node] = param.d.reshape(inshape)

    return alpha_normal, alpha_reduction


def parse_weights(args, alpha):
    ops_list = dict()
    num_intermediate_nodes = args.num_nodes - 2 - 1
    for i in range(num_intermediate_nodes):
        gene = list()
        W = alpha[i + 2]
        edges = sorted(range(i + 2),
                       key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != 7))[:2]

        for j in edges:
            k_best = None
            for k in range(len(W[j])):
                if k != 7:
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            gene.append((k_best, j))
        ops_list[i + 2] = gene
    return ops_list


def cutout(image, args):
    """
        cutout by numpy implementation.
    """
    h, w = image.shape[2:]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - args.cutout_length // 2, 0, h)
    y2 = np.clip(y + args.cutout_length // 2, 0, h)
    x1 = np.clip(x - args.cutout_length // 2, 0, w)
    x2 = np.clip(x + args.cutout_length // 2, 0, w)
    mask[y1: y2, x1: x2] = 0.
    image *= mask
    return image


def show_derived_cell(args, ops, architecture, celltype):
    """
        showing architecture configuration.
        celltype is either "normal" or "reduction".
    """
    if celltype == "normal":
        print("Normal Cell Architecture:")
    else:
        print("Reduction Cell Architecture:")
    print("Consists of {} nodes".format(args.num_nodes))
    print("ops at node 0:\n          None (Input node)")
    print("ops at node 1:\n          None (Input node)")
    for k, v in architecture.items():
        print("ops at node {}:".format(k))
        for elem in v:
            print("          {} from node {}".format(
                ops[elem[0]].__name__, elem[1]))
    print("ops at node {}:\n          Concatenation (Output node)\n".format(
        args.num_nodes - 1))


def learning_rate_scheduler(curr_iter, T_max, eta_max, eta_min=0):
    """
        cosine annealing scheduler.
    """
    lr = eta_min + 0.5 * (eta_max - eta_min) * \
        (1 + np.cos(np.pi*(curr_iter / T_max)))
    return lr


def categorical_error(pred, label):
    """
        Compute categorical error given score vectors and labels as
        numpy.ndarray.
    """
    pred_label = pred.argmax(1)
    return (pred_label != label.flat).mean()


def construct_aux_head(y, num_class):
    """
        Following the original implementation, we assume input size is 8x8.
    """
    h = F.relu(y)
    # size gets resized to 2 x 2
    h = F.average_pooling(h, kernel=(5, 5), stride=(3, 3), pad=(0, 0))
    with nn.parameter_scope("aux_head_0"):
        h = PF.convolution(h, 128, (1, 1), with_bias=False)
        h = PF.batch_normalization(h)
        h = F.relu(h)
    with nn.parameter_scope("aux_head_1"):
        h = PF.convolution(h, 768, (2, 2), with_bias=False)
        h = PF.batch_normalization(h)
        h = F.relu(h)
    with nn.parameter_scope("aux_head_out"):
        logit = PF.affine(h, num_class, with_bias=False)

    return logit


def conv1x1(x, output_filter, scope, test, is_search):
    """
        1x1 convolution (to adjust the feature maps between cells).
    """
    h = F.relu(x)
    with nn.parameter_scope(scope):
        h = PF.convolution(h, output_filter, (1, 1), with_bias=False)
        h = PF.batch_normalization(h, batch_stat=not test,
                                   fix_parameters=is_search)
    return h


def loss_function(pred, label, aux_logits=None, aux_weights=1.0):
    """
        Compute loss.
    """
    if aux_logits is None:
        loss = F.mean(F.softmax_cross_entropy(pred, label))
    else:
        loss = F.softmax_cross_entropy(pred, label)
        loss_from_aux = F.mul_scalar(
                    F.softmax_cross_entropy(aux_logits, label), aux_weights)
        loss = F.mean(F.add2(loss, loss_from_aux))
    return loss
