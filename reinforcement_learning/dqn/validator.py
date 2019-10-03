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
import os
import numpy as np

from nnabla.monitor import MonitorSeries


class Validator(object):

    def __init__(self, env, replay_memory, explorer, obs_sampler, num_episodes=10, clip_episode_step=False, num_ces=4500*4, num_eval_steps=125000*4, render=False, monitor=None, tbw=None):
        self.env = env
        self.replay_memory = replay_memory
        self.obs_sampler = obs_sampler
        self.explorer = explorer
        self.num_episodes = num_episodes
        self.clip_episode_step = clip_episode_step
        self.num_ces = num_ces
        self.num_eval_steps = num_eval_steps
        self.render = render
        self.steps = 0
        self.losses = []
        self.start_time = time.time()
        self.monitor = None
        if monitor:
            monitor_val_score = MonitorSeries("Validation Score", monitor)
            monitor_eval_score = MonitorSeries("Eval Score", monitor)
            self.monitor = {
                'val_score': monitor_val_score,
                'eval_score': monitor_eval_score
            }
        self.tbw = tbw

    def step_episode(self):
        # A. initialize env
        obs = self.env.reset()
        self.replay_memory.add_st(obs, False)

        if self.clip_episode_step:
            self.steps = 0

        # B. Time steps loop
        total_reward = 0.0
        while True:
            if self.render:
                self.env.render()
            # B-1. determine action
            obs_net = self.obs_sampler(self.replay_memory)[np.newaxis]
            action = self.explorer.select_action(obs_net, val=True)
            # B-2. one step forward simulator
            obs, reward, done, _ = self.env.step(action[0])
            self.replay_memory.add_ar(action, reward)
            total_reward += reward

            # B-. Increment
            self.steps += 1

            if self.clip_episode_step and self.steps >= self.num_ces:
                self.replay_memory.add_st(obs, True)
                break
            else:
                self.replay_memory.add_st(obs, done)

            # B-. Break if it's terminal
            if done:
                break

        return total_reward

    def step(self, cur_train_steps=-1):
        total_rewards = []
        for episode in range(self.num_episodes):
            total_rewards += [self.step_episode()]
            #print('episode{} ended in {} steps'.format(str(episode), self.steps))
        mean_reward = np.mean(total_rewards)
        if cur_train_steps >= 0:
            if self.monitor:
                self.monitor['val_score'].add(cur_train_steps, mean_reward)
            if self.tbw:
                self.tbw.add_scalar('validation/score',
                                    mean_reward, cur_train_steps)
        return mean_reward

    # This evaluation is mainly for DQN atari evaluation
    def evaluate(self, cur_train_steps):
        self.steps = 0
        total_rewards = []
        while True:
            reward = self.step_episode()
            if self.steps > self.num_eval_steps:
                mean_reward = np.mean(total_rewards)
                if self.monitor:
                    self.monitor['eval_score'].add(
                        cur_train_steps, mean_reward)
                if self.tbw:
                    self.tbw.add_scalar('evaluation/score',
                                        mean_reward, cur_train_steps)
                break
            else:
                total_rewards += [reward]
