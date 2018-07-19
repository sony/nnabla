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


class Trainer(object):

    def __init__(self, env, replay_memory, learner, sampler, explorer, obs_sampler, num_episodes=10, train_start=1000, train_freq=4, batch_size=32, render=False):
        self.env = env
        self.replay_memory = replay_memory
        self.learner = learner
        self.sampler = sampler
        self.obs_sampler = obs_sampler
        self.explorer = explorer
        self.num_episodes = num_episodes
        self.train_start = train_start
        self.train_started = False
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.render = render
        self.steps = 0
        self.losses = []

    def step_train(self):
        if self.replay_memory.seen < self.train_start:
            return

        if (self.steps + 1) % self.train_freq != 0:
            return

        # print("Train step at {}".format(self.steps))
        batch = self.replay_memory.sample(self.batch_size, self.sampler)
        loss = self.learner.update(batch)
        self.losses += [loss]
        self.train_started = True

    def step_episode(self):
        # A. initialize env
        obs = self.env.reset()
        self.replay_memory.add_st(obs, False)

        # B. Time steps loop
        total_reward = 0.0
        while True:
            if self.render:
                self.env.render()
            # B-1. determine action
            obs_net = self.obs_sampler(self.replay_memory)[np.newaxis]
            action = self.explorer.sample_action(obs_net)
            if self.train_started:
                self.explorer.update()

            # B-2. one step forward simulator
            obs, reward, done, _ = self.env.step(action[0])
            self.replay_memory.add_ar(action, reward)
            self.replay_memory.add_st(obs, done)
            total_reward += reward

            # B-3. Train step
            self.step_train()

            # B-. Increment
            self.steps += 1

            # B-. Break if it's terminal
            if done:
                break
        return total_reward

    def step(self):
        total_rewards = []
        for _ in range(self.num_episodes):
            total_rewards += [self.step_episode()]
        print('Mean reward: {}, Loss: {}, Epsilon: {}'.format(
            np.mean(total_rewards), np.mean(self.losses), self.explorer.epsilon()))
        self.losses = []
