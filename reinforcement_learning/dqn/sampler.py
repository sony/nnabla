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


class Sampler(object):
    def __init__(self, num_frames=4):
        self.num_frames = num_frames

    def __call__(self, replay_memory, i):
        if replay_memory.terminal[i] or replay_memory.cursor == i:
            return None
        inds = np.arange(i - self.num_frames + 1, i + 2)
        t_flags = replay_memory.terminal.take(inds, axis=0, mode='wrap')
        obs = replay_memory.obs.take(inds, axis=0, mode='wrap')
        action = replay_memory.action[i]
        reward = replay_memory.reward[i]
        terminal = t_flags[-1]
        if np.all(t_flags[:-1] == 0):
            return obs[:-1], action, reward, terminal, obs[1:]
        first = np.cumsum(t_flags[:-1]).argmax() + 1
        obs[:first] = obs[first]
        return obs[:-1], action, reward, terminal, obs[1:]


class ObsSampler(object):
    def __init__(self, num_frames=4):
        self.num_frames = num_frames

    def __call__(self, replay_memory):
        i = replay_memory.cursor
        if self.num_frames == 1:
            return replay_memory.obs[i][np.newaxis]
        inds = np.arange(i - self.num_frames + 1, i + 1)
        obs = replay_memory.obs.take(inds, axis=0, mode='wrap')
        t_flags = replay_memory.terminal.take(inds, axis=0, mode='wrap')
        if np.all(t_flags[:-1] == 0):
            return obs
        first = np.cumsum(t_flags[:-1]).argmax() + 1
        obs[:first] = obs[first]
        return obs
