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
import random
import os
import sys
from subprocess import call
import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solver as S
import nnabla.initializer as I

from args import get_args


class LSTMWrapper(PF.LSTMCell, object):
    def __init__(self, batch_size, state_size, h=None, c=None):
        super(LSTMWrapper, self).__init__(batch_size, state_size, h, c)
        self.h0 = self.h
        self.c0 = self.c
        self.h0.data.zero()
        self.c0.data.zero()

    def share_data(self):
        '''
        Initial cells point to the data of last cells so that learning continues
        '''
        self.h0.data = self.h.data
        self.c0.data = self.c.data


def gradient_clipping(params, max_norm, norm_type=2):
    params = list(filter(lambda p: p.need_grad == True, params))
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(np.abs(p.g).max() for p in params)
    else:
        total_norm = 0.
        for p in params:
            param_norm = F.pow_scalar(
                F.sum(p.grad ** norm_type), 1. / norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coeff = max_norm / (float(total_norm.data) + 1e-6)
    if clip_coeff < 1:
        for p in params:
            p.g = p.g * clip_coeff


def perplexity(loss):
    perplexity = np.exp(loss)
    return perplexity


def get_data():
    fnames = ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']
    for fname in fnames:
        if not os.path.exists(fname):
            url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/'+fname
            call(['wget', url])
    train_words = open('ptb.train.txt').read().replace('\n', '<eos>').split()
    words_as_set = set(train_words)
    word_to_id = {w: i for i, w in enumerate(words_as_set)}
    train_data = [word_to_id[w] for w in train_words]
    val_words = open('ptb.valid.txt').read().replace('\n', '<eos>').split()
    val_data = [word_to_id[w] for w in val_words]
    test_words = open('ptb.test.txt').read().replace('\n', '<eos>').split()
    test_data = [word_to_id[w] for w in val_words]

    return train_data, val_data, test_data


def get_batch(data, itr, bs, num_steps):
    offsets = [i * (len(data)-num_steps) // bs for i in range(bs)]
    cur_words = [data[(offset + itr) % (len(data)-num_steps):(offset + itr) %
                      (len(data)-num_steps) + num_steps] for offset in offsets]
    next_words = [data[(offset + itr) % (len(data)-num_steps) + 1:(offset + itr) %
                       (len(data)-num_steps) + num_steps + 1] for offset in offsets]
    return np.array(cur_words).reshape([bs, num_steps]), np.array(next_words).reshape([bs, num_steps])


def get_loss(l1, l2, x, t, w_init, b_init, num_words, batch_size, state_size, dropout=False, dropout_rate=0.5, embed_name='embed', pred_name='pred'):
    e_list = [PF.embed(x_elm, num_words, state_size, name=embed_name)
              for x_elm in F.split(x, axis=1)]
    t_list = F.split(t, axis=1)
    loss = 0
    for i, (e_t, t_t) in enumerate(zip(e_list, t_list)):
        if dropout:
            h1 = l1(F.dropout(e_t, dropout_rate), w_init, b_init)
            h2 = l2(F.dropout(h1, dropout_rate), w_init, b_init)
            y = PF.affine(F.dropout(h2, dropout_rate),
                          num_words, name=pred_name)
        else:
            h1 = l1(e_t, w_init, b_init)
            h2 = l2(h1, w_init, b_init)
            y = PF.affine(h2, num_words, name=pred_name)
        t_t = F.reshape(t_t, [batch_size, 1])
        loss += F.mean(F.softmax_cross_entropy(y, t_t))
    loss /= float(i+1)

    return loss


def main():

    args = get_args()
    state_size = args.state_size
    batch_size = args.batch_size
    num_steps = args.num_steps
    num_layers = args.num_layers
    max_epoch = args.max_epoch
    max_norm = args.gradient_clipping_max_norm
    num_words = 10000
    lr = args.learning_rate

    train_data, val_data, test_data = get_data()

    # Get context.
    from nnabla.ext_utils import get_extension_context
    logger.info("Running in %s" % args.context)
    ctx = get_extension_context(
        args.context, device_id=args.device_id, type_config=args.type_config)
    nn.set_default_context(ctx)

    from nnabla.monitor import Monitor, MonitorSeries
    monitor = Monitor(args.work_dir)
    monitor_perplexity = MonitorSeries(
        "Training perplexity", monitor, interval=10)
    monitor_vperplexity = MonitorSeries("Validation perplexity", monitor, interval=(
        len(val_data)//(num_steps*batch_size)))
    monitor_tperplexity = MonitorSeries(
        "Test perplexity", monitor, interval=(len(test_data)//(num_steps*1)))

    l1 = LSTMWrapper(batch_size, state_size)
    l2 = LSTMWrapper(batch_size, state_size)

    # train graph

    x = nn.Variable((batch_size, num_steps))
    t = nn.Variable((batch_size, num_steps))
    w = I.UniformInitializer((-0.1, 0.1))
    b = I.ConstantInitializer(1)
    loss = get_loss(l1, l2, x, t, w, b, num_words,
                    batch_size, state_size, True)
    l1.share_data()
    l2.share_data()

    # validation graph

    vx = nn.Variable((batch_size, num_steps))
    vt = nn.Variable((batch_size, num_steps))
    vloss = get_loss(l1, l2, vx, vt, w, b, num_words, batch_size, state_size)
    solver = S.Sgd(lr)
    solver.set_parameters(nn.get_parameters())

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    best_val = 10000
    for epoch in range(max_epoch):
        l1.reset_state()
        l2.reset_state()
        for i in range(len(train_data)//(num_steps*batch_size)):
            x.d, t.d = get_batch(train_data, i*num_steps,
                                 batch_size, num_steps)
            solver.zero_grad()
            loss.forward()
            loss.backward(clear_buffer=True)
            solver.weight_decay(1e-5)
            gradient_clipping(nn.get_parameters().values(), max_norm)
            solver.update()
            perp = perplexity(loss.d.copy())
            monitor_perplexity.add(
                (len(train_data)//(num_steps*batch_size))*(epoch)+i, perp)
        l1.reset_state()
        l2.reset_state()
        vloss_avg = 0
        for i in range(len(val_data)//(num_steps * batch_size)):
            vx.d, vt.d = get_batch(val_data, i*num_steps,
                                   batch_size, num_steps)
            vloss.forward()
            vloss_avg += vloss.d.copy()
        vloss_avg /= float((len(val_data)//(num_steps*batch_size)))
        vper = perplexity(vloss_avg)

        if vper < best_val:
            best_val = vper
            if vper < 200:
                save_name = "params_epoch_{:02d}.h5".format(epoch)
                nn.save_parameters(os.path.join(args.save_dir, save_name))
        else:
            solver.set_learning_rate(solver.learning_rate()*0.25)
            logger.info("Decreased learning rate to {:05f}".format(
                solver.learning_rate()))
        monitor_vperplexity.add(
            (len(val_data)//(num_steps*batch_size))*(epoch)+i, vper)

    # for final test split
    t_batch_size = 1
    tl1 = LSTMWrapper(t_batch_size, state_size)
    tl2 = LSTMWrapper(t_batch_size, state_size)
    tloss_avg = 0
    tx = nn.Variable((t_batch_size, num_steps))
    tt = nn.Variable((t_batch_size, num_steps))
    tloss = get_loss(tl1, tl2, tx, tt, w, b, num_words, 1, state_size)

    tl1.share_data()
    tl2.share_data()

    for i in range(len(test_data)//(num_steps * t_batch_size)):
        tx.d, tt.d = get_batch(test_data, i*num_steps, 1, num_steps)
        tloss.forward()
        tloss_avg += tloss.d.copy()
    tloss_avg /= float((len(test_data)//(num_steps*t_batch_size)))
    tper = perplexity(tloss_avg)
    monitor_tperplexity.add(
        (len(test_data)//(num_steps*t_batch_size))*(epoch)+i, tper)


if __name__ == '__main__':
    main()
