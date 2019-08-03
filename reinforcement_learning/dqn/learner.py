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

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from collections import namedtuple
import os


def update_variable_dict(d, h, name):
    if d is None:
        return
    d[name] = h


def _q_cnn(inp, convs, hiddens, num_actions, test=False, variable_dict=None):
    h = inp
    # Build convs
    for i, (o, k, s) in enumerate(convs):
        name = 'conv{}'.format(i + 1)
        h = F.relu(PF.convolution(h, o, (k, k), stride=(
            s, s), fix_parameters=test, name=name))
        update_variable_dict(variable_dict, h, name)
    # Build affines
    for i, o in enumerate(hiddens):
        name = 'fc{}'.format(i + 1)
        h = F.relu(PF.affine(h, o, fix_parameters=test, name=name))
        update_variable_dict(variable_dict, h, name)

    return PF.affine(h, num_actions, fix_parameters=test, name='fc_fin')


def q_cnn(inp, num_actions, test=False):
    return _q_cnn(
        inp,
        [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        [256],
        num_actions, test, None)


def q_mlp(s, num_actions, test=False):
    h = s
    for i, n in enumerate([64]):
        with nn.parameter_scope('fc%d' % (i + 1)):
            h = PF.affine(h, n, fix_parameters=test)
            h = F.relu(h)
    return PF.affine(h, num_actions, name='fc_fin', fix_parameters=test)


class QLearner(object):
    def __init__(self, q_builder, num_actions, clip_reward=1.0, sync_freq=500,
                 save_freq=10000, save_path=None,
                 gamma=0.99, learning_rate=5e-4, weight_decay=0, name_q='q', name_qnext='qnext'):
        self.built = False
        self.q_builder = q_builder
        self.num_actions = num_actions
        self.clip_reward = clip_reward
        self.sync_freq = sync_freq
        self.save_freq = save_freq
        if save_path is None:
            import output_path
            save_path = output_path.default_output_path()
        self.save_path = save_path
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.name_q = name_q
        self.name_qnext = name_qnext

        # count of neural network update (steps / train_freq)
        self.update_count = 0

    def build_train_graph(self, batch):
        self.solver = S.Adam(self.learning_rate)

        obs, action, reward, terminal, newobs = batch
        # Create input variables
        s = nn.Variable(obs.shape)
        a = nn.Variable(action.shape)
        r = nn.Variable(reward.shape)
        t = nn.Variable(terminal.shape)
        snext = nn.Variable(newobs.shape)
        with nn.parameter_scope(self.name_q):
            q = self.q_builder(s, self.num_actions, test=False)
            self.solver.set_parameters(nn.get_parameters())
        with nn.parameter_scope(self.name_qnext):
            qnext = self.q_builder(snext, self.num_actions, test=True)
        qnext.need_grad = False
        clipped_r = F.minimum_scalar(F.maximum_scalar(
            r, -self.clip_reward), self.clip_reward)
        q_a = F.sum(
            q * F.one_hot(F.reshape(a, (-1, 1), inplace=False), (q.shape[1],)), axis=1)
        target = clipped_r + self.gamma * (1 - t) * F.max(qnext, axis=1)
        loss = F.mean(F.huber_loss(q_a, target))
        Variables = namedtuple(
            'Variables', ['s', 'a', 'r', 't', 'snext', 'q', 'loss'])
        self.v = Variables(s, a, r, t, snext, q, loss)
        self.sync_models()
        self.built = True

    def sync_models(self):
        with nn.parameter_scope(self.name_q):
            q_params = nn.get_parameters(grad_only=False)
        with nn.parameter_scope(self.name_qnext):
            qnext_params = nn.get_parameters(grad_only=False)
        for k, v in q_params.items():
            qnext_params[k].data.copy_from(v.data)

    def save_model(self):
        from nnabla.utils.save import save
        with nn.parameter_scope(self.name_q):
            save(self.save_path.get_filepath('qnet_{:08d}.nnp'.format(self.update_count)),
                 {'networks': [
                     {'name': 'qnet',
                      'batch_size': self.v.s.shape[0],
                      'outputs': {'q': self.v.q},
                      'names': {'s': self.v.s}}
                 ]
            })

    def update(self, batch):
        if not self.built:
            self.build_train_graph(batch)
        self.v.s.d = batch[0]
        self.v.a.d = batch[1]
        self.v.r.d = batch[2]
        self.v.t.d = batch[3]
        self.v.snext.d = batch[4]
        self.solver.zero_grad()
        self.v.loss.forward(clear_no_need_grad=True)
        self.v.loss.backward(clear_buffer=True)
        if self.weight_decay:
            self.solver.weight_decay(self.weight_decay)
        self.solver.update()
        self.update_count += 1
        if self.update_count % self.sync_freq == 0:
            self.sync_models()
        if self.update_count % self.save_freq == 0:
            self.save_model()
        return self.v.loss.d.copy()
