
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


class ReplayMemory(object):

    def __init__(self, obs_dims, action_dims, max_memory=1000000,
                 obs_dtype=np.float32, action_dtype=np.float32,
                 reward_dtype=np.float32):
        self.obs = np.empty((max_memory,) + obs_dims, dtype=obs_dtype)
        self.action = np.empty((max_memory,) + action_dims, dtype=action_dtype)
        self.reward = np.empty((max_memory,), dtype=reward_dtype)
        self.terminal = np.ones((max_memory,), dtype=np.uint8)
        self.add_st_called = False
        self.cursor = -1
        self.seen = -1

    def add_st(self, obs, terminal):
        assert not self.add_st_called, "calling add_s consecutively is prohibited."
        self.cursor += 1
        self.seen += 1
        if self.cursor >= self.reward.shape[0]:
            self.cursor = 0
        # print("cursor={}, seen={}".format(self.cursor, self.seen))
        self.obs[self.cursor] = obs
        self.terminal[self.cursor] = terminal
        if terminal:
            # Do not toggle add_st_called because add_st is called
            # at the beginning of the next episode.
            return
        self.add_st_called = True

    def add_ar(self, action, reward):
        assert self.add_st_called, "add_art must be called after add_s is called."
        self.action[self.cursor] = action
        self.reward[self.cursor] = reward
        self.add_st_called = False

    def sample(self, num_samples, sampler, rng=np.random):
        max_ind = min(max(self.seen, self.cursor), self.obs.shape[0])
        obs = []
        action = []
        reward = []
        terminal = []
        newobs = []
        i = 0
        while i < num_samples:
            j = rng.randint(max_ind)
            sample = sampler(self, j)
            if sample is None:
                continue
            obs += [sample[0]]
            action += [sample[1]]
            reward += [sample[2]]
            terminal += [sample[3]]
            newobs += [sample[4]]
            i += 1
        obs = np.stack(obs)
        action = np.stack(action)
        reward = np.stack(reward)
        terminal = np.stack(terminal)
        newobs = np.stack(newobs)
        return obs, action, reward, terminal, newobs
