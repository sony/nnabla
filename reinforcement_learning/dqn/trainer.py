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

import numpy as np
import time

from nnabla.monitor import MonitorSeries


class Trainer(object):

    def __init__(self, env, replay_memory, learner, sampler, explorer,
                 obs_sampler, inter_eval_steps=-1, num_episodes=10,
                 train_start=1000, train_freq=4, batch_size=32, render=False,
                 validator=None, monitor=None, tbw=None):
        self.env = env
        self.replay_memory = replay_memory
        self.learner = learner
        self.sampler = sampler
        self.explorer = explorer
        self.obs_sampler = obs_sampler
        self.inter_eval_steps = inter_eval_steps
        self.num_episodes = num_episodes
        self.train_start = train_start
        self.train_started = False
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.render = render
        self.validator = validator
        self.steps = 0
        self.losses = []
        self.monitor = None
        if monitor:
            monitor_loss = MonitorSeries("QNetwork Loss", monitor)
            monitor_epsilon = MonitorSeries("Epsilon", monitor)
            monitor_train_reward = MonitorSeries("Train Reward", monitor)
            self.monitor = {
                'loss': monitor_loss,
                'epsilon': monitor_epsilon,
                'reward': monitor_train_reward
            }
        self.tbw = tbw

    def trained_steps(self):
        return self.steps-self.train_start

    def step_train(self):
        batch = self.replay_memory.sample(self.batch_size, self.sampler)
        loss = self.learner.update(batch)
        self.losses += [loss]
        self.train_started = True

    def step_episode(self):
        # A. initialize env
        obs = self.env.reset()
        self.replay_memory.add_st(obs, False)

        # B. Time steps(=frames) loop
        total_reward = 0.0
        while True:
            if self.render:
                self.env.render()

            # B-1. determine action
            obs_net = self.obs_sampler(self.replay_memory)[np.newaxis]
            action = self.explorer.select_action(obs_net)
            if self.train_started:
                self.explorer.update()  # for epsilon linear decay

            # B-2. one step forward simulator
            obs, reward, done, _ = self.env.step(action[0])
            self.replay_memory.add_ar(action, reward)
            self.replay_memory.add_st(obs, done)
            total_reward += reward

            # B-3. Train step
            if (self.steps+1) > self.train_start and \
               (self.steps+1) % self.train_freq == 0:
                self.step_train()

            # B-. Increment
            self.steps += 1

            # B-. Force evaluate in a certain number of intermediate step
            if self.trained_steps() != 0 and \
               self.trained_steps() % self.inter_eval_steps == 0:
                self.validator.evaluate(self.trained_steps())

            # B-. Break if it's terminal
            if done:
                break

        return total_reward

    def step(self):
        total_rewards = []
        for _ in range(self.num_episodes):
            total_rewards += [self.step_episode()]

        # log output
        if self.trained_steps() >= 0:
            if self.monitor:
                self.monitor['loss'].add(self.trained_steps(),
                                         np.mean(self.losses))
                self.monitor['reward'].add(self.trained_steps(),
                                           np.mean(total_rewards))
                self.monitor['epsilon'].add(self.trained_steps(),
                                            self.explorer.epsilon)
            if self.tbw:
                self.tbw.add_scalar('training/loss',
                                    np.mean(self.losses), self.trained_steps())
                self.tbw.add_scalar('training/reward',
                                    np.mean(total_rewards), self.trained_steps())
                self.tbw.add_scalar('training/epsilon',
                                    self.explorer.epsilon, self.trained_steps())

        # validate if validator is available
        if self.validator:
            self.validator.step(self.trained_steps())

        self.losses = []
