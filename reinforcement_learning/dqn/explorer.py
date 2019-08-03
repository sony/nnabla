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

from abc import ABC, abstractmethod


class Explorer(ABC):
    def __init__(self, num_actions, name='q', network=None):
        self.num_actions = num_actions
        self.name = name
        self.network = network

    @abstractmethod
    def select_action(self, obs, val=False):
        raise NotImplementedError()

    def build_network(self, obs):
        s = nn.Variable(obs.shape)
        with nn.parameter_scope(self.name):
            q = self.q_builder(s, self.num_actions, test=True)
        Variables = namedtuple(
            'Variables', ['s', 'q'])
        self.network = Variables(s, q)

    def build_network_from_nnp(self, nnp_file):
        from nnabla.utils import nnp_graph
        nnp = nnp_graph.NnpLoader(nnp_file)
        net = nnp.get_network(self.name, batch_size=1)
        s = net.inputs['s']
        q = net.outputs['q']
        Variables = namedtuple(
            'Variables', ['s', 'q'])
        assert q.shape[1] == self.num_actions
        self.network = Variables(s, q)


class GreedyExplorer(Explorer):
    def __init__(self, num_actions, use_nnp=False,
                 q_builder=None, nnp_file=None, name='q'):
        self.num_actions = num_actions
        self.q_builder = q_builder
        self.network = None
        self.name = name
        if use_nnp:
            if nnp_file is None:
                return
            super().build_network_from_nnp(nnp_file)

    def select_action(self, obs, val=False):
        if not self.network:
            super().build_network(obs)
        self.network.s.d = obs
        self.network.q.forward(clear_buffer=True)
        return np.argmax(self.network.q.d, axis=1)


class EGreedyExplorer(Explorer):
    def __init__(self, num_actions, epsilon=0.0, use_nnp=False,
                 q_builder=None, nnp_file=None, name='q', rng=np.random):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.q_builder = q_builder
        self.network = None
        self.name = name
        self.rng = rng
        if use_nnp:
            if nnp_file is None:
                return
            super().build_network_from_nnp(nnp_file)

    def select_action(self, obs, val=False):
        if not self.network:
            super().build_network(obs)
        self.network.s.d = obs
        self.network.q.forward(clear_buffer=True)
        if self.rng.rand() >= self.epsilon:
            return np.argmax(self.network.q.d, axis=1)
        return self.rng.randint(self.num_actions, size=(obs.shape[0],))


class LinearDecayEGreedyExplorer(EGreedyExplorer):
    def __init__(self, num_actions, eps_val=0.05, eps_start=.9, eps_end=.1, eps_steps=1e6,
                 use_nnp=False, q_builder=None, nnp_file=None,
                 name='q', rng=np.random):
        super().__init__(num_actions, epsilon=eps_start, use_nnp=use_nnp,
                         q_builder=q_builder, nnp_file=nnp_file,
                         name=name, rng=rng)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.eps_val = eps_val
        self.time = 0

    def update(self):
        self.time += 1

    def linear_decay_epsilon(self):
        self.epsilon = max(
           self.eps_end,
           self.eps_start +
           (self.eps_end - self.eps_start) * self.time / self.eps_steps)

    def select_action(self, obs, val=False):
        if val:
            self.epsilon = self.eps_val
        else:
            self.linear_decay_epsilon()
        return super().select_action(obs)
