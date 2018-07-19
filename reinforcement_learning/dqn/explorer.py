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

from collections import namedtuple

import numpy as np

import nnabla as nn


class NnpExplorer(object):
    def __init__(self, nnp_file, num_actions, name='qnet', rng=np.random):
        self.num_actions = num_actions
        self.rng = rng
        if nnp_file is None:
            self.network = None
            return
        from nnabla.utils import nnp_graph
        nnp = nnp_graph.NnpLoader(nnp_file)
        self.network = nnp.get_network(name, batch_size=1)
        assert self.network.outputs['q'].shape[1] == num_actions

    def select_greedy_action(self, obs):
        if self.network is None:
            return np.array([self.rng.randint(self.num_actions)])
        s = self.network.inputs['s']
        q = self.network.outputs['q']
        s.d = obs
        q.forward(clear_buffer=True)
        return np.argmax(q.d, axis=1)


class EpsilonGreedyQExplorer(object):
    def __init__(self, q_builder, num_actions, eps_start=.9, eps_end=.1, eps_steps=10**6, name='q', rng=np.random):
        self.built = False
        self.q_builder = q_builder
        self.num_actions = num_actions
        self.name = name
        self.rng = rng
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.time = 0

    def build_graph(self, obs):
        s = nn.Variable(obs.shape)
        with nn.parameter_scope(self.name):
            q = self.q_builder(s, self.num_actions, test=True)
        Variables = namedtuple(
            'Variables', ['s', 'q'])
        self.v = Variables(s, q)
        self.built = True

    def sample_action(self, obs):
        if not self.built:
            self.build_graph(obs)
        self.v.s.d = obs
        self.v.q.forward(clear_buffer=True)

        if self.rng.rand() >= self.epsilon():
            return np.argmax(self.v.q.d, axis=1)
        return self.rng.randint(self.num_actions, size=(obs.shape[0],))

    def select_greedy_action(self, obs):
        if not self.built:
            self.build_graph(obs)
        self.v.s.d = obs
        self.v.q.forward(clear_buffer=True)
        return np.argmax(self.v.q.d, axis=1)

    def epsilon(self):
        return max(
            self.eps_end,
            self.eps_start + (self.eps_end - self.eps_start) * self.time / self.eps_steps)

    def update(self):
        self.time += 1
