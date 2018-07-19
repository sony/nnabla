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

import time
import numpy as np


class Validator(object):

    def __init__(self, env, replay_memory, explorer, obs_sampler, num_episodes=10, render=False, log_path=None):
        self.env = env
        self.replay_memory = replay_memory
        self.obs_sampler = obs_sampler
        self.explorer = explorer
        self.num_episodes = num_episodes
        self.render = render
        if log_path is None:
            import output_path
            log_path = output_path.default_output_path()
        elif not log_path:
            self.log_path = False
        if log_path:
            self.log_path = log_path.get_filepath('reward.txt')
            with open(self.log_path, 'w'):
                pass
        self.steps = 0
        self.losses = []
        self.start_time = time.time()

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
            action = self.explorer.select_greedy_action(obs_net)
            # B-2. one step forward simulator
            obs, reward, done, _ = self.env.step(action[0])
            self.replay_memory.add_ar(action, reward)
            self.replay_memory.add_st(obs, done)
            total_reward += reward

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
        mean_reward = np.mean(total_rewards)
        print('Mean test reward: {}'.format(mean_reward))
        if self.log_path:
            with open(self.log_path, 'a') as fd:
                print(time.time() - self.start_time, mean_reward, file=fd)
