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


def lr_scheduler(curr_iter, T_max, eta_max, eta_min=0):
    """
        cosine annealing scheduler.
    """
    lr = eta_min + 0.5 * (eta_max - eta_min) * \
        (1 + np.cos(np.pi*(curr_iter / T_max)))
    return lr


def update_learning_rate_cosine(current_iter, eta_max, eta_min, n_devices):
    """
       restart cosine learning rate function after every quarter of tatal iteration.
    """
    N_iter = 1000000//4//n_devices  # 1Mil iterations used by original authors
    restart = [250000//n_devices, 500000//n_devices, 750000//n_devices]
    if current_iter <= restart[0]:
        current_lr = lr_scheduler(current_iter, N_iter, eta_max, eta_min)
        return current_lr
    if restart[0] < current_iter <= restart[1]:
        current_lr = lr_scheduler(
            current_iter-restart[0], N_iter, eta_max, eta_min)
        return current_lr
    if restart[1] < current_iter <= restart[2]:
        current_lr = lr_scheduler(
            current_iter - restart[1], N_iter, eta_max, eta_min)
        return current_lr
    if current_iter > restart[2]:
        current_lr = lr_scheduler(
            current_iter - restart[2], N_iter, eta_max, eta_min)
        return current_lr


def update_learning_rate_multistep(current_iter, lr_steps, lr):
    if current_iter in lr_steps:
        lr *= 0.5
    return lr


if __name__ == "__main__":
    """
        Draw graph to see how cosine annealing rate affects the learning rate.
    """
    lr_l = []

    #########################################################################
    # for cosine annealing learnig rate scheduler
    n_devices = 4
    train_size = 32208//16//n_devices
    total_epochs = 497

    #restart = [250000//n_devices,500000//n_devices,750000//n_devices]
    eta_max = 2e-4
    eta_min = 1e-7
    i = 0
    for epoch in range(total_epochs):
        index = 0
        while (index < train_size):
            i += 1
            lr = update_learning_rate_cosine(i, eta_max, eta_min, n_devices)
            lr_l.append(lr)
            index += 1
    #########################################################################
    # for multistep learnig rate scheduler
    # lr = 1e-4
    # n_devices = 4
    # lr_steps = [50000//n_devices, 100000//n_devices, 200000//n_devices, 300000//n_devices]
    # total_epochs = 199
    # N_iter = 400000/n_devices
    # train_size = 32208 // 16 // n_devices
    # i = 0
    # for epoch in range(199):
    #     index = 0
    #     while (index < train_size):
    #         i+=1
    #         lr = update_learning_rate_multistep(i, lr_steps, lr)
    #         lr_l.append(lr)
    #         index+=1
    ##########################################################################
    # to visualize the LR scheduler
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick

    mpl.style.use('default')
    import seaborn
    seaborn.set(style='whitegrid')
    seaborn.set_context('paper')

    plt.figure(1)
    plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title('Nnabla', fontsize=16, color='k')
    plt.plot(list(range(train_size*total_epochs)), lr_l,
             linewidth=1.5, label='learning rate scheme')
    legend = plt.legend(loc='upper right', shadow=False)
    ax = plt.gca()
    labels = ax.get_xticks().tolist()
    for k, v in enumerate(labels):
        labels[k] = str(int(v / 1000)) + 'K'
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    ax.set_ylabel('Learning rate')
    ax.set_xlabel('Iteration')
    fig = plt.gcf()
    plt.show()
